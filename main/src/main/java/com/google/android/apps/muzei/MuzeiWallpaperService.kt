/*
 * Copyright 2014 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.android.apps.muzei

import android.annotation.SuppressLint
import android.app.WallpaperColors
import android.app.WallpaperManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.ViewConfiguration
import android.view.WindowInsets
import android.view.WindowManager
import androidx.annotation.RequiresApi
import androidx.core.content.ContextCompat
import androidx.core.os.UserManagerCompat
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.apps.muzei.featuredart.BuildConfig.FEATURED_ART_AUTHORITY
import com.google.android.apps.muzei.notifications.NotificationUpdater
import com.google.android.apps.muzei.render.ImageLoader
import com.google.android.apps.muzei.render.relativeLuminance
import com.google.android.apps.muzei.render.MuzeiBlurRenderer
import com.google.android.apps.muzei.render.RealRenderController
import com.google.android.apps.muzei.render.RenderController
import com.google.android.apps.muzei.room.Artwork
import com.google.android.apps.muzei.room.MuzeiDatabase
import com.google.android.apps.muzei.room.contentUri
import com.google.android.apps.muzei.room.openArtworkInfo
import com.google.android.apps.muzei.settings.EffectsLockScreenOpen
import com.google.android.apps.muzei.settings.Prefs
import com.google.android.apps.muzei.shortcuts.ArtworkInfoShortcutController
import com.google.android.apps.muzei.sync.ProviderManager
import com.google.android.apps.muzei.util.collectIn
import com.google.android.apps.muzei.wallpaper.LockscreenObserver
import com.google.android.apps.muzei.wallpaper.WallpaperAnalytics
import com.google.android.apps.muzei.wearable.WearableController
import com.google.android.apps.muzei.widget.WidgetUpdater
import com.google.firebase.Firebase
import com.google.firebase.analytics.FirebaseAnalytics
import com.google.firebase.analytics.analytics
import com.google.firebase.analytics.logEvent
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import net.rbgrn.android.glwallpaperservice.GLWallpaperService

data class WallpaperSize(val width: Int, val height: Int)

val WallpaperSizeStateFlow = MutableStateFlow<WallpaperSize?>(null)

class MuzeiWallpaperService : GLWallpaperService(), LifecycleOwner {

    companion object {
        private const val TEMPORARY_FOCUS_DURATION_MILLIS: Long = 3000
        private const val THREE_FINGER_TAP_INTERVAL_MS = 1000L
        // Resolution we decode the artwork at to extract colours. Larger than WallpaperColors
        // strictly needs (~112px) so the thin status-bar strip we crop out still has some
        // vertical resolution to average over.
        private const val COLOR_DECODE_SIZE = 256 // px
        // Fallback status bar height if the platform dimen can't be resolved.
        private const val DEFAULT_STATUS_BAR_HEIGHT_DP = 24f
        // Mean relative luminance (gamma-corrected, 0 = black .. 1 = white) the status-bar strip
        // must reach for the system to use dark icons — our tunable replacement for fromBitmap()'s
        // non-tunable internal calculation. Same metric the framework uses, where its (stricter,
        // also dark-pixel-guarded) threshold is ~0.70. Higher => dark icons only over brighter
        // artwork. (Only applied on API 31+, where the WallpaperColors hint can be set explicitly.)
        private const val STATUS_BAR_DARK_ICON_MIN_LUMINANCE = 0.42f
    }

    private val wallpaperLifecycle = LifecycleRegistry(this)
    private var unlockReceiver: BroadcastReceiver? = null

    override fun onCreateEngine(): Engine {
        return MuzeiWallpaperEngine()
    }

    @SuppressLint("InlinedApi", "WrongConstant")
    override fun onCreate() {
        super.onCreate()
        with(wallpaperLifecycle) {
            addObserver(WorkManagerInitializer.initializeObserver(this@MuzeiWallpaperService))
            addObserver(NotificationUpdater(this@MuzeiWallpaperService))
            addObserver(WearableController(this@MuzeiWallpaperService))
            addObserver(WidgetUpdater(this@MuzeiWallpaperService))
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N_MR1) {
                addObserver(ArtworkInfoShortcutController(this@MuzeiWallpaperService))
            }
        }
        ProviderManager.getInstance(this).observe(this) { provider ->
            if (provider == null) {
                lifecycleScope.launch {
                    withContext(NonCancellable) {
                        ProviderManager.select(this@MuzeiWallpaperService, FEATURED_ART_AUTHORITY)
                    }
                }
            }
        }
        if (UserManagerCompat.isUserUnlocked(this)) {
            wallpaperLifecycle.handleLifecycleEvent(Lifecycle.Event.ON_START)
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            unlockReceiver = object : BroadcastReceiver() {
                override fun onReceive(context: Context, intent: Intent) {
                    wallpaperLifecycle.handleLifecycleEvent(Lifecycle.Event.ON_START)
                    unregisterReceiver(this)
                    unlockReceiver = null
                }
            }
            val filter = IntentFilter(Intent.ACTION_USER_UNLOCKED)
            ContextCompat.registerReceiver(
                this,
                unlockReceiver,
                filter,
                ContextCompat.RECEIVER_NOT_EXPORTED
            )
        }
    }

    override val lifecycle: Lifecycle = wallpaperLifecycle

    override fun onDestroy() {
        if (unlockReceiver != null) {
            unregisterReceiver(unlockReceiver)
        }
        wallpaperLifecycle.handleLifecycleEvent(Lifecycle.Event.ON_DESTROY)
        super.onDestroy()
    }

    inner class MuzeiWallpaperEngine
        : GLEngine(),
            LifecycleOwner,
            DefaultLifecycleObserver,
            RenderController.Callbacks,
            MuzeiBlurRenderer.Callbacks {

        private lateinit var renderer: MuzeiBlurRenderer
        private lateinit var renderController: RenderController
        private var currentArtworkColors: WallpaperColors? = null
        // notifyColorsChanged() only causes a launcher offset jump when the home launcher is
        // hosting the visible wallpaper. It's deferred until then and flushed once we reach a safe
        // state — surface hidden, or the lock screen (keyguard) hosting (see flushPendingColors).
        private var surfaceVisible = false
        private var lockScreenVisible = false
        private var pendingColorsChanged = false
        // The dark-text hint we last published — state for the hint-flip notification gate in
        // updateCurrentArtwork(). We only extract WallpaperColors to drive the status-bar icon
        // colour (HINT_SUPPORTS_DARK_TEXT), so we notify only when that hint flips — never for
        // colour-only changes. Each notifyColorsChanged() that reaches Nova Launcher costs it
        // memory (it leaks per wallpaper-colours callback), so suppressing every notification that
        // wouldn't change the hint is the main lever we have to stop walking it toward an OOM.
        // Delete this field together with that gate to notify on every change.
        private var lastNotifiedHints: Int = Int.MIN_VALUE

        private var validDoubleTap: Boolean = false
        private var lastThreeFingerTap = 0L

        private val engineLifecycle = LifecycleRegistry(this)

        private var doubleTapTimeout: Job? = null

        private val gestureListener = object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent): Boolean {
                return true
            }

            override fun onDoubleTap(e: MotionEvent): Boolean {
                if (ArtDetailOpen.value) {
                    // The main activity is visible, so discard any double touches since focus
                    // should be forced on
                    return true
                }

                validDoubleTap = true // processed in onCommand/COMMAND_TAP

                doubleTapTimeout?.cancel()
                val timeout = ViewConfiguration.getDoubleTapTimeout().toLong()
                doubleTapTimeout = lifecycleScope.launch {
                    delay(timeout)
                    queueEvent {
                        validDoubleTap = false
                    }
                }
                return true
            }
        }
        private val gestureDetector: GestureDetector = GestureDetector(this@MuzeiWallpaperService,
                gestureListener)

        private var delayedBlur: Job? = null

        override fun onCreate(surfaceHolder: SurfaceHolder) {
            super<GLEngine>.onCreate(surfaceHolder)

            renderer = MuzeiBlurRenderer(this@MuzeiWallpaperService, this,
                    false, isPreview)
            renderController = RealRenderController(this@MuzeiWallpaperService,
                    renderer, this)
            engineLifecycle.addObserver(renderController)
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 0, 0, 0)
            setRenderer(renderer)
            renderMode = RENDERMODE_WHEN_DIRTY
            requestRender()

            engineLifecycle.handleLifecycleEvent(Lifecycle.Event.ON_CREATE)
            engineLifecycle.addObserver(WallpaperAnalytics(this@MuzeiWallpaperService))
            engineLifecycle.addObserver(LockscreenObserver(this@MuzeiWallpaperService, this))
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                lifecycleScope.launch {
                    repeatOnLifecycle(Lifecycle.State.STARTED) {
                        val database = MuzeiDatabase.getInstance(this@MuzeiWallpaperService)
                        database.artworkDao().getCurrentArtworkFlow()
                            .filterNotNull().distinctUntilChanged().collectLatest { artwork ->
                                updateCurrentArtwork(artwork)
                            }
                    }
                }

            }

            // Use the MuzeiWallpaperService's lifecycle to wait for the user to unlock
            wallpaperLifecycle.addObserver(this)
            setTouchEventsEnabled(true)
            setOffsetNotificationsEnabled(true)
            EffectsLockScreenOpen.collectIn(this) { isEffectsLockScreenOpen ->
                renderController.onLockScreen = isEffectsLockScreenOpen
            }
            ArtDetailOpen.collectIn(this) { isArtDetailOpened ->
                cancelDelayedBlur()
                queueEvent { renderer.setIsBlurred(!isArtDetailOpened, true) }
            }

            ArtDetailViewport.getChanges().collectIn(this) {
                requestRender()
            }
        }

        override val lifecycle: Lifecycle = engineLifecycle

        override fun onStart(owner: LifecycleOwner) {
            // The MuzeiWallpaperService only gets to ON_START when the user is unlocked
            // At that point, we can proceed with the engine's lifecycle
            // In preview mode, we only move to ON_START to avoid analytics events.
            engineLifecycle.handleLifecycleEvent(if (isPreview)
                Lifecycle.Event.ON_START else Lifecycle.Event.ON_RESUME)
        }

        @RequiresApi(Build.VERSION_CODES.O_MR1)
        private suspend fun updateCurrentArtwork(artwork: Artwork) {
            val stripFraction = statusBarStripFraction()
            currentArtworkColors = withContext(Dispatchers.IO) {
                val image = ImageLoader.decode(
                        contentResolver, artwork.contentUri, COLOR_DECODE_SIZE)
                        ?: return@withContext null
                // Derive WallpaperColors — and in particular the HINT_SUPPORTS_DARK_TEXT flag
                // that drives the status bar icon colour — from just the strip of the artwork
                // sitting below the status bar, rather than the whole image. The artwork is
                // cover-fit, so the top of the image lines up with the top of the screen.
                val stripHeight = (image.height * stripFraction).toInt().coerceIn(1, image.height)
                val strip = Bitmap.createBitmap(image, 0, 0, image.width, stripHeight)
                val base = WallpaperColors.fromBitmap(strip)
                val colors = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    // fromBitmap()'s own dark-text calculation isn't tunable, so decide the hint
                    // ourselves from the strip's mean relative luminance and rebuild with the real
                    // colours. (No dark-pixel-area guard, unlike the framework — mean only.)
                    val luminance = strip.relativeLuminance()
                    val hints = if (luminance >= STATUS_BAR_DARK_ICON_MIN_LUMINANCE) {
                        WallpaperColors.HINT_SUPPORTS_DARK_TEXT
                    } else {
                        0
                    }
                    WallpaperColors(base.primaryColor, base.secondaryColor, base.tertiaryColor, hints)
                } else {
                    base
                }
                if (strip != image) strip.recycle()
                image.recycle()
                colors
            } ?: return
            // --- hint-flip notification gate: delete this whole block to notify on every change ---
            // Notify only when the dark-text hint flips. Colours change on every artwork but we
            // don't act on them, so a notifyColorsChanged() that wouldn't change the hint is pure
            // waste that, repeated, leaks Nova Launcher toward an OOM. onComputeColors() still
            // returns the latest colours, so the next real hint flip publishes them anyway.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val hints = currentArtworkColors?.colorHints ?: 0
                if (hints == lastNotifiedHints) {
                    return
                }
                lastNotifiedHints = hints
            }
            // --- end hint-flip notification gate ---
            // The launcher reacts to notifyColorsChanged() by re-pushing wallpaper offsets — a
            // visible jump we can't filter out — but only while it's hosting the visible home
            // wallpaper. That's the case only when the surface is visible, not on the lock screen
            // (keyguard hosts), and not with the Muzei app foreground (its window hosts via
            // windowShowWallpaper). In every other state it's safe to notify now.
            if (surfaceVisible && !lockScreenVisible && !MuzeiActivityVisible.value) {
                pendingColorsChanged = true
            } else {
                notifyColorsChanged()
            }
        }

        /**
         * Fraction of the wallpaper height occupied by the status bar, used to crop the strip
         * of artwork whose luminance decides the status bar icon colour.
         */
        private fun statusBarStripFraction(): Float {
            val res = resources
            // Prefer the real status bar inset (accounts for cutouts, foldables, per-display
            // differences) where available; fall back to a dp estimate below API 30.
            val statusBarHeight = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                getSystemService(WindowManager::class.java).currentWindowMetrics
                        .windowInsets.getInsets(WindowInsets.Type.statusBars()).top
                        .takeIf { it > 0 }
            } else {
                null
            } ?: (DEFAULT_STATUS_BAR_HEIGHT_DP * res.displayMetrics.density).toInt()
            val screenHeight = WallpaperSizeStateFlow.value?.height
                    ?: res.displayMetrics.heightPixels
            return if (screenHeight > 0) {
                (statusBarHeight.toFloat() / screenHeight).coerceIn(0.01f, 0.5f)
            } else {
                0.05f
            }
        }

        @RequiresApi(Build.VERSION_CODES.O_MR1)
        override fun onComputeColors(): WallpaperColors? =
            currentArtworkColors ?: super.onComputeColors()

        /** Publishes a deferred colours update. Only call from a state where notifyColorsChanged()
         *  won't cause the launcher offset jump (surface hidden, or lock screen hosting). */
        private fun flushPendingColors() {
            if (pendingColorsChanged) {
                pendingColorsChanged = false
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                    notifyColorsChanged()
                }
            }
        }

        override fun onSurfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
            super.onSurfaceChanged(holder, format, width, height)
            if (!isPreview) {
                WallpaperSizeStateFlow.value = WallpaperSize(width, height)
            }
            renderController.reloadCurrentArtwork()
        }

        override fun onDestroy() {
            wallpaperLifecycle.removeObserver(this)
            engineLifecycle.handleLifecycleEvent(Lifecycle.Event.ON_DESTROY)
            queueEvent {
                renderer.destroy()
            }
            super<GLEngine>.onDestroy()
        }

        /**
         * Pause video playback whenever it isn't actually on screen — the surface is hidden (screen
         * off / AOD / another app) or the lock screen is showing — and resume it only when the home
         * wallpaper is visible. Images are unaffected (a no-op for them).
         */
        private fun updateVideoPlaybackState() {
            renderController.setVideoPlaybackPaused(!surfaceVisible || lockScreenVisible)
        }

        fun lockScreenVisibleChanged(isLockScreenVisible: Boolean) {
            lockScreenVisible = isLockScreenVisible
            updateVideoPlaybackState()
            if (isLockScreenVisible) {
                // The keyguard (not the launcher) hosts the wallpaper now, so notifyColorsChanged()
                // here won't cause the launcher offset jump — flush any deferred colours update.
                flushPendingColors()
            }
            // Crossfades that start while Muzei isn't the visible surface (e.g. unlocking to
            // an app rather than the home screen) used to stall and flicker on resume. That is
            // now handled by keeping the engine rendering in the background (onVisibilityChanged)
            // and deferring the animator's onEnd until after the final frame is drawn
            // (TickingFloatAnimator / MuzeiBlurRenderer.onDrawFrame).
            if (!EffectsLockScreenOpen.value) {
                renderController.onLockScreen = isLockScreenVisible
            }
        }

        override fun onVisibilityChanged(visible: Boolean) {
            // Keep the renderer "visible" regardless of the actual surface visibility so that
            // in-flight crossfade/blur animations keep ticking to completion even while Muzei
            // is in the background (e.g. behind the lock screen or another app). Without this,
            // a transition started off-screen stalls and flickers when Muzei next resumes.
            // RENDERMODE_WHEN_DIRTY means this only costs frames while an animation is actually
            // running — an idle background wallpaper still doesn't render.
            renderController.visible = true

            surfaceVisible = visible
            // Pause video while it isn't actually on screen (hidden surface or lock screen) so it
            // stops decoding and requesting frames; resume when the home wallpaper is shown.
            updateVideoPlaybackState()
            if (!visible) {
                // Hidden now (screen off / another app) — safe to publish a deferred update.
                flushPendingColors()
            }
        }

        override fun onOffsetsChanged(
                xOffset: Float,
                yOffset: Float,
                xOffsetStep: Float,
                yOffsetStep: Float,
                xPixelOffset: Int,
                yPixelOffset: Int
        ) {
            super.onOffsetsChanged(xOffset, yOffset, xOffsetStep, yOffsetStep, xPixelOffset,
                    yPixelOffset)
            // Only the launcher home screen should drive the wallpaper pan. The keyguard and the
            // in-app screens (art detail, and the Sources/Effects tabs behind it) also host the
            // wallpaper and push meaningless offsets that would overwrite the home-screen pan and make
            // it jump — so ignore offsets while either owns the wallpaper, keeping the last home pan.
            // (xOffset itself is fine at 0 — the left edge — so we gate on who's hosting, not on the
            // offset or its step; that also honours a single-page launcher, which reports step 0.)
            if (lockScreenVisible || MuzeiActivityVisible.value) {
                return
            }
            renderer.setNormalOffsetX(xOffset)
        }

        override fun onZoomChanged(zoom: Float) {
            super.onZoomChanged(zoom)
            renderer.setZoom(zoom)
        }

        override fun onCommand(
                action: String?,
                x: Int,
                y: Int,
                z: Int,
                extras: Bundle?,
                resultRequested: Boolean
        ): Bundle? {
            // validDoubleTap previously set in the gesture listener
            if (WallpaperManager.COMMAND_TAP == action && validDoubleTap) {
                val prefs = Prefs.getSharedPreferences(this@MuzeiWallpaperService)
                val doubleTapValue = prefs.getString(Prefs.PREF_DOUBLE_TAP,
                        null) ?: Prefs.PREF_TAP_ACTION_TEMP
                triggerTapAction(doubleTapValue, "gesture_double_tap")
                // Reset the flag
                validDoubleTap = false
            }
            return super.onCommand(action, x, y, z, extras, resultRequested)
        }

        private fun triggerTapAction(action: String, type: String) {
            when (action) {
                Prefs.PREF_TAP_ACTION_TEMP -> {
                    Firebase.analytics.logEvent("temp_disable_effects") {
                        param(FirebaseAnalytics.Param.CONTENT_TYPE, type)
                    }
                    // Temporarily toggle focused/blurred
                    queueEvent {
                        renderer.setIsBlurred(!renderer.isBlurred, false)
                        // Schedule a re-blur
                        delayedBlur()
                    }
                }
                Prefs.PREF_TAP_ACTION_NEXT -> {
                    lifecycleScope.launch {
                        withContext(NonCancellable) {
                            Firebase.analytics.logEvent("next_artwork") {
                                param(FirebaseAnalytics.Param.CONTENT_TYPE, type)
                            }
                            ProviderManager.getInstance(this@MuzeiWallpaperService).nextArtwork()
                        }
                    }
                }
                Prefs.PREF_TAP_ACTION_VIEW_DETAILS -> {
                    lifecycleScope.launch {
                        withContext(NonCancellable) {
                            val artwork = MuzeiDatabase
                                .getInstance(this@MuzeiWallpaperService)
                                .artworkDao()
                                .getCurrentArtwork()
                            artwork?.run {
                                Firebase.analytics.logEvent("artwork_info_open") {
                                    param(FirebaseAnalytics.Param.CONTENT_TYPE, type)
                                }
                                openArtworkInfo(this@MuzeiWallpaperService)
                            }
                        }
                    }
                }
            }
        }

        override fun onTouchEvent(event: MotionEvent) {
            super.onTouchEvent(event)
            gestureDetector.onTouchEvent(event)
            // Delay blur from temporary refocus while touching the screen
            delayedBlur()
            // See if there was a valid three finger tap
            val now = SystemClock.elapsedRealtime()
            val timeSinceLastThreeFingerTap = now - lastThreeFingerTap
            if (event.pointerCount == 3
                && timeSinceLastThreeFingerTap > THREE_FINGER_TAP_INTERVAL_MS) {
                lastThreeFingerTap = now
                val prefs = Prefs.getSharedPreferences(this@MuzeiWallpaperService)
                val threeFingerTapValue = prefs.getString(Prefs.PREF_THREE_FINGER_TAP,
                        null) ?: Prefs.PREF_TAP_ACTION_NONE

                triggerTapAction(threeFingerTapValue, "gesture_three_finger")
            }
        }

        private fun cancelDelayedBlur() {
            delayedBlur?.cancel()
        }

        private fun delayedBlur() {
            if (ArtDetailOpen.value || renderer.isBlurred) {
                return
            }

            cancelDelayedBlur()
            delayedBlur = lifecycleScope.launch {
                delay(TEMPORARY_FOCUS_DURATION_MILLIS)
                queueEvent {
                    renderer.setIsBlurred(isBlurred = true, artDetailMode = false)
                }
            }
        }

        override fun queueEventOnGlThread(event: () -> Unit) {
            queueEvent {
                event()
            }
        }
    }
}