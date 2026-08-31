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
import androidx.core.os.UserManagerCompat
import androidx.lifecycle.LifecycleOwner
import com.google.android.apps.muzei.api.MuzeiContract
import com.google.android.apps.muzei.room.MuzeiDatabase
import com.google.android.apps.muzei.room.contentUri
import com.google.android.apps.muzei.util.collectIn
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.withContext

class RealRenderController(
        context: Context,
        renderer: MuzeiBlurRenderer,
        callbacks: Callbacks
) : RenderController(context, renderer, callbacks) {

    /**
     * If there's no artwork yet (as is the case when in Direct Boot), then we
     * use [MuzeiContract.Artwork.CONTENT_URI].
     */
    private var currentArtworkUri = MuzeiContract.Artwork.CONTENT_URI

    override fun onCreate(owner: LifecycleOwner) {
        super.onCreate(owner)
        // Direct Boot only: the artwork DB isn't readable while the user is locked, so fall back to
        // the generic CONTENT_URI (which reads the downloaded file directly). Once unlocked, onStart's
        // artwork flow loads the specific per-artwork URI; doing both loads the same artwork under two
        // different URIs, and since the video dedup keys on the raw URI it doesn't skip the second —
        // producing a video-to-identical-video crossfade. Skip the redundant load when unlocked.
        if (!UserManagerCompat.isUserUnlocked(context)) {
            reloadCurrentArtwork()
        }
    }

    override fun onStart(owner: LifecycleOwner) {
        super.onStart(owner)
        val database = MuzeiDatabase.getInstance(context)
        database.artworkDao().getCurrentArtworkFlow().filterNotNull()
                // Provider row updates (e.g. supportsNextArtwork being refreshed by
                // ProviderChangedWorker right after a provider switch) re-emit the same
                // artwork. Without this, switching providers decodes the image twice and
                // restarts the crossfade mid-animation, which reads as lag.
                .distinctUntilChanged()
                .collectIn(owner) { artwork ->
            currentArtworkUri = artwork.contentUri
            reloadCurrentArtwork()
        }
    }

    override suspend fun openDownloadedCurrentArtwork(): RenderSource {
        // The media type isn't stored, so probe the artwork's container each reload to tell video
        // from a still image. videoMimeType opens the file and runs MediaMetadataRetriever, so keep
        // it off the main thread.
        val uri = currentArtworkUri
        val isVideo = withContext(Dispatchers.IO) {
            context.contentResolver.videoMimeType(uri) != null
        }
        return if (isVideo) {
            RenderSource.Video(uri)
        } else {
            RenderSource.Image(ContentUriImageLoader(context.contentResolver, uri))
        }
    }
}