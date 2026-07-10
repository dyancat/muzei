/*
 * Copyright 2018 Google Inc.
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

package com.google.android.apps.muzei.browse

import android.app.Application
import android.content.ContentUris
import android.database.ContentObserver
import android.net.Uri
import android.os.DeadObjectException
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.viewModelScope
import com.google.android.apps.muzei.room.Artwork
import com.google.android.apps.muzei.room.getInstalledProviders
import com.google.android.apps.muzei.util.ContentProviderClientCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.FlowPreview
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.debounce
import kotlinx.coroutines.flow.emptyFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.shareIn
import kotlinx.coroutines.flow.withIndex
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

@OptIn(ExperimentalCoroutinesApi::class, FlowPreview::class)
class BrowseProviderViewModel(
    application: Application,
    savedStateHandle: SavedStateHandle
): AndroidViewModel(application) {

    private val args = BrowseProviderFragmentArgs.fromSavedStateHandle(savedStateHandle)

    val providerInfo = getInstalledProviders(application)
        .debounceExceptFirst(PROVIDER_CHANGE_DEBOUNCE)
        .map { providers ->
            providers.firstOrNull { it.authority == args.contentUri.authority }
        }

    val client = providerInfo.map { providerInfo ->
        providerInfo?.let {
            val context = getApplication<Application>()
            ContentProviderClientCompat.getClient(context, args.contentUri)
        }
    }

    private fun getProviderArtwork(
        contentProviderClient: ContentProviderClientCompat,
    ) = callbackFlow {
        val context = getApplication<Application>()
        val authority: String = args.contentUri.authority!!
        var refreshJob: Job? = null
        val refreshArt = {
            refreshJob?.cancel()
            // Query + cursor parsing is a binder IPC plus per-row object creation; keep it off
            // the main thread so opening the screen doesn't block on it.
            refreshJob = launch(Dispatchers.IO) {
                try {
                    val list = mutableListOf<BrowseArtwork>()
                    contentProviderClient.query(args.contentUri)?.use { data ->
                        while(data.moveToNext() && isActive) {
                            val providerArtwork =
                                com.google.android.apps.muzei.api.provider.Artwork.fromCursor(data)
                            val artwork = Artwork(ContentUris.withAppendedId(args.contentUri,
                                providerArtwork.id)).apply {
                                title = providerArtwork.title
                                byline = providerArtwork.byline
                                attribution = providerArtwork.attribution
                                providerAuthority = authority
                            }
                            // Read the name/size/file-modified time the grid can be sorted by;
                            // the added date is the artwork's own dateAdded.
                            val attrs = readSortAttributes(context, providerArtwork)
                            list.add(BrowseArtwork(
                                artwork = artwork,
                                name = attrs.name,
                                dateAdded = runCatching { providerArtwork.dateAdded.time }.getOrNull(),
                                dateModified = attrs.dateModified,
                                size = attrs.size,
                            ))
                        }
                    }
                    send(list)
                } catch (_: DeadObjectException) {
                    // Provider was updated out from underneath us
                    // so there's nothing more we can do here
                }
            }
        }
        val contentObserver = object : ContentObserver(null) {
            override fun onChange(selfChange: Boolean, uri: Uri?) {
                refreshArt()
            }
        }
        context.contentResolver.registerContentObserver(
                args.contentUri,
                true,
                contentObserver)
        refreshArt()

        awaitClose {
            context.contentResolver.unregisterContentObserver(contentObserver)
            contentProviderClient.close()
        }
    }

    private val browseArtwork = client.flatMapLatest { client ->
        if (client != null) {
            getProviderArtwork(client) }
        else {
            emptyFlow()
        }
    }.shareIn(viewModelScope, SharingStarted.WhileSubscribed(5000L), 1)

    private val _sortOption = MutableStateFlow(
        BrowseSortOption.fromPreferences(application))

    /**
     * The current browse-grid sort option. Persisted across sessions.
     */
    val sortOption = _sortOption

    fun setSortOption(option: BrowseSortOption) {
        getApplication<Application>().browseSharedPreferences().let { option.writeTo(it) }
        _sortOption.value = option
    }

    val artwork = combine(browseArtwork, _sortOption) { artwork, option ->
        option.sort(artwork).map { it.artwork }
    }.shareIn(viewModelScope, SharingStarted.WhileSubscribed(5000L), 1)

    companion object {
        // Coalesce the burst of package broadcasts during a provider install/update, but let the
        // initial (already-current) provider list through immediately so the screen loads at once.
        private const val PROVIDER_CHANGE_DEBOUNCE = 1000L

        /**
         * Like [debounce], but emits the first value immediately instead of waiting out the
         * timeout. A plain [debounce] delays every emission, so it held back the initial provider
         * list for a full second on every screen open, showing an empty grid until it elapsed.
         */
        private fun <T> Flow<T>.debounceExceptFirst(timeoutMillis: Long): Flow<T> =
            withIndex()
                .debounce { (index, _) -> if (index == 0) 0L else timeoutMillis }
                .map { it.value }
    }
}