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

package com.google.android.apps.muzei.browse

import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.provider.DocumentsContract
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Log
import androidx.core.content.edit
import androidx.core.net.toUri
import com.google.android.apps.muzei.room.Artwork
import com.google.android.apps.muzei.api.provider.Artwork as ProviderArtwork

private const val TAG = "BrowseSortOption"

private const val SHARED_PREF_NAME = "BrowseProvider"
private const val PREF_SORT_CRITERION = "sort_criterion"
private const val PREF_SORT_DESCENDING = "sort_descending"

/**
 * The attribute the browse grid can be sorted by.
 */
enum class BrowseSortCriterion {
    NAME,
    DATE_ADDED,
    DATE_MODIFIED,
    SIZE,
}

/**
 * A chosen sort criterion plus its direction.
 */
data class BrowseSortOption(
    val criterion: BrowseSortCriterion,
    val descending: Boolean,
) {
    /**
     * Order [artwork] according to this option. Items whose attribute couldn't be read sort to
     * the end regardless of direction; ties break on the artwork id so the order is stable.
     */
    fun sort(artwork: List<BrowseArtwork>): List<BrowseArtwork> {
        val comparator: Comparator<BrowseArtwork> = when (criterion) {
            BrowseSortCriterion.NAME -> {
                val order = if (descending) {
                    String.CASE_INSENSITIVE_ORDER.reversed()
                } else {
                    String.CASE_INSENSITIVE_ORDER
                }
                compareBy(nullsLast(order)) { it.name }
            }
            BrowseSortCriterion.DATE_ADDED -> compareBy(directionalNullsLast()) { it.dateAdded }
            BrowseSortCriterion.DATE_MODIFIED -> compareBy(directionalNullsLast()) { it.dateModified }
            BrowseSortCriterion.SIZE -> compareBy(directionalNullsLast()) { it.size }
        }
        return artwork.sortedWith(comparator.thenBy { it.artwork.id })
    }

    private fun <T : Comparable<T>> directionalNullsLast(): Comparator<T?> =
        nullsLast(if (descending) reverseOrder() else naturalOrder())

    fun writeTo(preferences: SharedPreferences) {
        preferences.edit {
            putString(PREF_SORT_CRITERION, criterion.name)
            putBoolean(PREF_SORT_DESCENDING, descending)
        }
    }

    companion object {
        /**
         * Default sort: newest added first, matching the ContentProvider's historical
         * `date_added DESC` default order for the browse grid.
         */
        val DEFAULT = BrowseSortOption(BrowseSortCriterion.DATE_ADDED, descending = true)

        fun fromPreferences(context: Context): BrowseSortOption {
            val preferences = context.browseSharedPreferences()
            val criterion = preferences.getString(PREF_SORT_CRITERION, null)
                ?.let { name -> runCatching { BrowseSortCriterion.valueOf(name) }.getOrNull() }
                ?: return DEFAULT
            return BrowseSortOption(
                criterion,
                preferences.getBoolean(PREF_SORT_DESCENDING, DEFAULT.descending),
            )
        }
    }
}

internal fun Context.browseSharedPreferences(): SharedPreferences =
    getSharedPreferences(SHARED_PREF_NAME, Context.MODE_PRIVATE)

/**
 * A browse-grid artwork paired with the attributes it can be sorted by. `dateAdded` is when the
 * artwork was added to the provider; `dateModified` is the source file's last-modified time.
 */
data class BrowseArtwork(
    val artwork: Artwork,
    val name: String?,
    val dateAdded: Long?,
    val dateModified: Long?,
    val size: Long?,
)

/**
 * The sortable attributes of a single artwork, read from its underlying URI.
 */
internal data class SortAttributes(
    val name: String? = null,
    val size: Long? = null,
    val dateModified: Long? = null,
)

/**
 * Read the sortable name, size and file modified time for [providerArtwork] from its underlying
 * URI. Best-effort: any attribute we can't read is left null (and sorts to the end). Safe to call
 * off the main thread. The added date comes from the artwork's `dateAdded`, so it isn't read here.
 */
internal fun readSortAttributes(
    context: Context,
    providerArtwork: ProviderArtwork,
): SortAttributes {
    val uri = providerArtwork.persistentUri
        ?: providerArtwork.webUri
        ?: providerArtwork.token?.takeUnless { it.isEmpty() }?.toUri()
    var name: String? = null
    var size: Long? = null
    var dateModified: Long? = null
    if (uri != null) {
        try {
            context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME).takeIf { it >= 0 }
                        ?.let { if (!cursor.isNull(it)) name = cursor.getString(it) }
                    cursor.getColumnIndex(OpenableColumns.SIZE).takeIf { it >= 0 }
                        ?.let { if (!cursor.isNull(it)) size = cursor.getLong(it) }
                    // SAF documents expose last_modified in milliseconds; MediaStore exposes
                    // date_modified in seconds instead, so normalise that to millis.
                    cursor.getColumnIndex(DocumentsContract.Document.COLUMN_LAST_MODIFIED)
                        .takeIf { it >= 0 }
                        ?.let { if (!cursor.isNull(it)) dateModified = cursor.getLong(it) }
                    if (dateModified == null) {
                        cursor.getColumnIndex(MediaStore.MediaColumns.DATE_MODIFIED)
                            .takeIf { it >= 0 }
                            ?.let { if (!cursor.isNull(it)) dateModified = cursor.getLong(it) * 1000 }
                    }
                }
            }
        } catch (e: Exception) {
            // Could be a SecurityException, IllegalArgumentException, remote URI, etc.
            Log.i(TAG, "Unable to read sort attributes for $uri", e)
        }
    }
    // Fall back to the locally-cached file for anything we couldn't read from the URI.
    val cachedFile = runCatching { providerArtwork.data }.getOrNull()
    if (cachedFile?.exists() == true) {
        if (name == null) name = cachedFile.name
        if (size == null) size = cachedFile.length()
        if (dateModified == null) cachedFile.lastModified().takeIf { it > 0 }?.let { dateModified = it }
    }
    return SortAttributes(name, size, dateModified)
}
