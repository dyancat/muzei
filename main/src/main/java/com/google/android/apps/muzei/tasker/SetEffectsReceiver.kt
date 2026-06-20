/*
 * Copyright 2026 Google Inc.
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

package com.google.android.apps.muzei.tasker

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import androidx.core.content.edit
import com.google.android.apps.muzei.settings.Prefs

/**
 * Exported receiver that lets external automation apps (e.g. Tasker, via its built-in
 * "Send Intent" action) change the blur, dim, and grey effect amounts at runtime.
 *
 * Fire an intent with action [ACTION_SET_EFFECTS] and any of the extras [EXTRA_BLUR],
 * [EXTRA_DIM], [EXTRA_GREY]. Each is given as a percentage from 0 to 100 (clamped) and is
 * mapped onto the effect's internal range. Only extras that are present are changed.
 *
 * The optional string extra [EXTRA_SCREEN] selects which screen is affected:
 * [SCREEN_HOME], [SCREEN_LOCK], or [SCREEN_BOTH] (the default).
 *
 * The wallpaper renderer ([com.google.android.apps.muzei.render.RenderController]) observes
 * these preference keys and re-renders automatically, so no further action is needed here.
 *
 * Example (Tasker → Send Intent), setting blur and dim to 50%:
 *   Action: net.nurik.roman.muzei.action.SET_EFFECTS
 *   Extras: blur:50, dim:50, screen:both
 *   Target: Broadcast Receiver
 */
class SetEffectsReceiver : BroadcastReceiver() {

    companion object {
        private const val ACTION_SET_EFFECTS = "net.nurik.roman.muzei.action.SET_EFFECTS"

        private const val EXTRA_BLUR = "blur"
        private const val EXTRA_DIM = "dim"
        private const val EXTRA_GREY = "grey"
        private const val EXTRA_SCREEN = "screen"

        private const val SCREEN_HOME = "home"
        private const val SCREEN_LOCK = "lock"
        private const val SCREEN_BOTH = "both"

        // Matches the slider ranges in EffectsScreen.kt
        private const val MAX_BLUR = 500
        private const val MAX_DIM = 255
        private const val MAX_GREY = 500
    }

    override fun onReceive(context: Context, intent: Intent?) {
        if (intent?.action != ACTION_SET_EFFECTS) {
            return
        }

        val screen = intent.getStringExtra(EXTRA_SCREEN)?.lowercase() ?: SCREEN_BOTH
        val affectHome = screen == SCREEN_HOME || screen == SCREEN_BOTH
        val affectLock = screen == SCREEN_LOCK || screen == SCREEN_BOTH

        val blur = intent.readEffectExtra(EXTRA_BLUR, MAX_BLUR)
        val dim = intent.readEffectExtra(EXTRA_DIM, MAX_DIM)
        val grey = intent.readEffectExtra(EXTRA_GREY, MAX_GREY)

        if (blur == null && dim == null && grey == null) {
            return
        }

        Prefs.getSharedPreferences(context).edit {
            if (blur != null) {
                if (affectHome) putInt(Prefs.PREF_BLUR_AMOUNT, blur)
                if (affectLock) putInt(Prefs.PREF_LOCK_BLUR_AMOUNT, blur)
            }
            if (dim != null) {
                if (affectHome) putInt(Prefs.PREF_DIM_AMOUNT, dim)
                if (affectLock) putInt(Prefs.PREF_LOCK_DIM_AMOUNT, dim)
            }
            if (grey != null) {
                if (affectHome) putInt(Prefs.PREF_GREY_AMOUNT, grey)
                if (affectLock) putInt(Prefs.PREF_LOCK_GREY_AMOUNT, grey)
            }
        }
    }

    /**
     * Reads an effect amount given as a 0-100 percentage and maps it onto `0..max`. The extra
     * may arrive as a number or, when sent from Tasker's "Send Intent" action, as a string.
     * Returns null when the extra is absent; the percentage is clamped to `0..100`.
     */
    private fun Intent.readEffectExtra(name: String, max: Int): Int? {
        if (!hasExtra(name)) {
            return null
        }
        val percent = when (val extra = extras?.get(name)) {
            is Number -> extra.toDouble()
            is String -> extra.trim().toDoubleOrNull() ?: return null
            else -> return null
        }
        return Math.round(percent.coerceIn(0.0, 100.0) / 100.0 * max).toInt()
    }
}
