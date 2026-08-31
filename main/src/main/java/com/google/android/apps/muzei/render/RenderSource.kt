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

import android.net.Uri

/**
 * The current artwork as the renderer needs to consume it: either a still [Image] (decoded to a
 * bitmap and blurred as before) or a [Video] (played live and blurred on the GPU per frame).
 *
 * This is the seam that lets [MuzeiBlurRenderer] treat video and image artwork uniformly — the
 * [RenderController] resolves which one the current artwork is (from its recorded MIME type) and
 * hands the renderer the right variant.
 */
sealed class RenderSource {
    /** A still image, loaded through the existing [ImageLoader] decode path. */
    class Image(val loader: ImageLoader) : RenderSource()

    /** A video, played from [uri] (a seekable content URI served by MuzeiProvider). */
    class Video(val uri: Uri) : RenderSource()

    /**
     * A stable key identifying the underlying artwork, used to detect a no-op reload/switch
     * (mirrors how the image path keyed off the [ImageLoader]'s URI string).
     */
    override fun toString(): String = when (this) {
        is Image -> loader.toString()
        is Video -> uri.toString()
    }
}
