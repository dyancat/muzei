/*
 * Copyright 2024 Google Inc.
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

package com.google.android.apps.muzei.render

import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Surface
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.DefaultLoadControl
import androidx.media3.exoplayer.ExoPlayer

/**
 * A single, process-wide [ExoPlayer] shared by every [GLVideo] (there is one [GLVideo] per wallpaper
 * engine — home, lock, and the in-app preview — plus a spare during a crossfade).
 *
 * Only one engine is ever on screen at a time, and a decoded frame only needs to reach that engine's
 * texture, so rather than each engine running its own hardware decoder (a second resident
 * MediaCodec + its DMA-BUF buffer ring, easily hundreds of MB), they all share this one player and
 * simply re-point its output [Surface] at whichever engine is currently visible via [bind]. Each
 * [GLVideo] keeps only its (cheap) [android.graphics.SurfaceTexture] + external texture; the hidden
 * engines hold no decoder at all.
 *
 * The [ExoPlayer] is not thread-safe and lives on the main thread; every method here hops there. The
 * mutable state ([player], [boundSink], [currentUri], [retainCount], [screenOn]) is therefore only
 * ever touched inside the posted runnables (i.e. on the main thread), so it needs no locking even
 * though callers invoke these from the GL thread ([GLVideo]) and the main thread (the service).
 */
internal object SharedVideoPlayer {

    private const val TAG = "SharedVideoPlayer"

    private val mainHandler = Handler(Looper.getMainLooper())

    private var player: ExoPlayer? = null
    // Number of live GLVideos referencing the player; the player is released when it hits zero.
    private var retainCount = 0
    // The video currently loaded into the player (deduped so switching engines for the same artwork
    // doesn't re-prepare it).
    private var currentUri: Uri? = null
    // The GLVideo whose Surface the player is currently rendering into (null when no engine is on
    // screen). Identity only, so it's typed as Any.
    private var boundSink: Any? = null
    // Global screen state; the player only decodes while the screen is on (and something is bound).
    private var screenOn = true

    /** Registers a live user of the shared player. Balanced by [release]. */
    fun retain() {
        mainHandler.post {
            retainCount++
        }
    }

    /** Drops a user of the shared player, releasing the underlying [ExoPlayer] when the last one goes. */
    fun release() {
        mainHandler.post {
            retainCount--
            if (retainCount <= 0) {
                retainCount = 0
                player?.release()
                player = null
                currentUri = null
                boundSink = null
            }
        }
    }

    /**
     * Makes [sink] the active output: ensures the player is loaded with [uri] and points its video
     * surface at [surface]. Called when an engine becomes the visible one (see
     * MuzeiBlurRenderer.updateVideoBinding). Creates the player on first use.
     */
    fun bind(sink: Any, uri: Uri, surface: Surface, appContext: Context) {
        mainHandler.post {
            val p = player ?: createPlayer(appContext).also { player = it }
            boundSink = sink
            if (currentUri != uri) {
                currentUri = uri
                p.setMediaItem(MediaItem.fromUri(uri))
                p.prepare()
            }
            p.setVideoSurface(surface)
            updatePlayWhenReady()
        }
    }

    /**
     * Releases [sink] from being the active output if it currently is (an engine going off screen, or
     * a GLVideo being torn down). Leaves the player alive for whichever engine binds next.
     */
    fun unbind(sink: Any) {
        mainHandler.post {
            if (boundSink === sink) {
                boundSink = null
                player?.clearVideoSurface()
                updatePlayWhenReady()
            }
        }
    }

    /** Sets whether the screen is on; the player pauses immediately when it goes off. */
    fun setScreenOn(on: Boolean) {
        mainHandler.post {
            screenOn = on
            updatePlayWhenReady()
        }
    }

    // The player runs only while an engine is bound (on screen) and the screen is on. Anything else
    // (screen off, or every engine hidden behind another app) stops it decoding.
    private fun updatePlayWhenReady() {
        player?.playWhenReady = boundSink != null && screenOn
    }

    private fun createPlayer(appContext: Context): ExoPlayer {
        // A silent, looping wallpaper clip needs almost no look-ahead, so cap buffering hard. The
        // default LoadControl buffers up to ~50s, which for a high-bitrate video is ~100MB of heap; a
        // ~1-2s buffer keeps it to a few MB.
        val loadControl = DefaultLoadControl.Builder()
                .setBufferDurationsMs(
                        /* minBufferMs = */ 1_000,
                        /* maxBufferMs = */ 2_000,
                        /* bufferForPlaybackMs = */ 250,
                        /* bufferForPlaybackAfterRebufferMs = */ 500)
                .setPrioritizeTimeOverSizeThresholds(true)
                .build()
        return ExoPlayer.Builder(appContext)
                .setLoadControl(loadControl)
                .build().apply {
                    repeatMode = Player.REPEAT_MODE_ONE // seamless loop
                    volume = 0f // wallpapers are silent
                    addListener(object : Player.Listener {
                        override fun onPlayerError(error: PlaybackException) {
                            Log.e(TAG, "Error playing video: ${error.errorCodeName}", error)
                        }
                    })
                }
    }
}
