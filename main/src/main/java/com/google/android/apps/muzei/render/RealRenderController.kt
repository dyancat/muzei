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

package com.google.android.apps.muzei.render

import android.content.Context
import android.net.Uri
import androidx.lifecycle.LifecycleOwner
import com.google.android.apps.muzei.api.MuzeiContract
import com.google.android.apps.muzei.room.MuzeiDatabase
import com.google.android.apps.muzei.room.Screen
import com.google.android.apps.muzei.room.contentUri
import com.google.android.apps.muzei.util.collectIn
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filterNotNull

class RealRenderController(
        context: Context,
        renderer: MuzeiBlurRenderer,
        callbacks: Callbacks
) : RenderController(context, renderer, callbacks) {

    /**
     * If there's no artwork yet (as is the case when in Direct Boot), then we
     * use [MuzeiContract.Artwork.CONTENT_URI].
     */
    private var homeArtworkUri: Uri = MuzeiContract.Artwork.CONTENT_URI
    /**
     * The lock screen's current artwork, or null when the lock screen is "linked"
     * to home (it has no provider of its own) so we fall back to [homeArtworkUri].
     */
    private var lockArtworkUri: Uri? = null
    private var currentArtworkUri = MuzeiContract.Artwork.CONTENT_URI

    override fun onCreate(owner: LifecycleOwner) {
        super.onCreate(owner)
        reloadCurrentArtwork()
    }

    override fun onStart(owner: LifecycleOwner) {
        super.onStart(owner)
        val database = MuzeiDatabase.getInstance(context)
        database.artworkDao().getCurrentArtworkFlow(Screen.HOME.value).filterNotNull()
                // Provider row updates (e.g. supportsNextArtwork being refreshed by
                // ProviderChangedWorker right after a provider switch) re-emit the same
                // artwork. Without this, switching providers decodes the image twice and
                // restarts the crossfade mid-animation, which reads as lag.
                .distinctUntilChanged()
                .collectIn(owner) { artwork ->
            homeArtworkUri = artwork.contentUri
            if (activeScreen == Screen.HOME) {
                updateCurrentArtwork()
            }
        }
        // The lock flow is intentionally not filtered for null: a null means the
        // lock screen is linked to home (no lock provider row), and we must react
        // to that transition to fall back to the home artwork.
        database.artworkDao().getCurrentArtworkFlow(Screen.LOCK.value)
                .distinctUntilChanged()
                .collectIn(owner) { artwork ->
            lockArtworkUri = artwork?.contentUri
            if (activeScreen == Screen.LOCK) {
                updateCurrentArtwork()
            }
        }
    }

    override fun onActiveScreenChanged(screen: Screen) {
        updateCurrentArtwork()
    }

    /**
     * Point [currentArtworkUri] at the active screen's artwork (lock falls back to
     * home when linked) and crossfade to it only if it actually changed. When the
     * lock screen is linked this resolves to the home artwork, so a lock<->home
     * transition is a no-op and stays performance-neutral.
     */
    private fun updateCurrentArtwork() {
        val targetUri = when (activeScreen) {
            Screen.HOME -> homeArtworkUri
            Screen.LOCK -> lockArtworkUri ?: homeArtworkUri
        }
        if (targetUri != currentArtworkUri) {
            currentArtworkUri = targetUri
            reloadCurrentArtwork()
        }
    }

    override suspend fun openDownloadedCurrentArtwork() =
            ContentUriImageLoader(context.contentResolver, currentArtworkUri)
}