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

package com.google.android.apps.muzei.sync

import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.ContentResolver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.database.ContentObserver
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.RemoteException
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.Observer
import androidx.preference.PreferenceManager
import com.google.android.apps.muzei.api.internal.ProtocolConstants
import com.google.android.apps.muzei.api.provider.ProviderContract
import com.google.android.apps.muzei.room.Artwork
import com.google.android.apps.muzei.room.MuzeiDatabase
import com.google.android.apps.muzei.room.Provider
import com.google.android.apps.muzei.room.Screen
import com.google.android.apps.muzei.util.ContentProviderClientCompat
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import net.nurik.roman.muzei.androidclientcommon.BuildConfig
import java.util.concurrent.Executors

/**
 * Single threaded coroutine context used for all sync operations
 */
internal val syncSingleThreadContext by lazy {
    Executors.newSingleThreadExecutor { target ->
        Thread(target, "ProviderSync")
    }.asCoroutineDispatcher()
}

/**
 * Manager which monitors the current Provider
 */
class ProviderManager private constructor(private val context: Context)
    : MutableLiveData<Provider?>(), Observer<Provider?> {

    /**
     * Enum that represents the order in which new artwork is loaded
     */
    enum class LoadOrdering {
        IN_ORDER,
        NEW_IN_ORDER,
        RANDOM,
    }

    companion object {
        private const val TAG = "ProviderManager"
        private const val PREF_LOAD_FREQUENCY_SECONDS = "loadFrequencySeconds"
        private const val DEFAULT_LOAD_FREQUENCY_SECONDS = 3600L
        private const val PREF_LOAD_ON_WIFI = "loadOnWifi"
        private const val DEFAULT_LOAD_ON_WIFI = false
        private const val PREF_LOAD_ORDERING = "loadOrdering"
        private val DEFAULT_LOAD_ORDERING = LoadOrdering.NEW_IN_ORDER

        @SuppressLint("StaticFieldLeak")
        @Volatile
        private var instance: ProviderManager? = null

        fun getInstance(context: Context): ProviderManager =
                instance ?: synchronized(this) {
                    instance ?: ProviderManager(context.applicationContext)
                            .also { instance = it }
                }

        suspend fun select(context: Context, authority: String, screen: Screen = Screen.HOME) {
            val providerDao = MuzeiDatabase.getInstance(context).providerDao()
            val currentAuthority = providerDao.getProvider(screen.value)?.authority
            if (authority != currentAuthority) {
                providerDao.select(authority, screen.value)
            }
        }

        /**
         * Remove the lock screen's provider selection, "linking" the lock screen
         * back to the home screen's provider.
         */
        suspend fun clearLock(context: Context) {
            MuzeiDatabase.getInstance(context).providerDao().clearLock()
        }

        suspend fun requestLoad(context: Context, contentUri: Uri) {
            try {
                ContentProviderClientCompat.getClient(context, contentUri)?.call(
                        ProtocolConstants.METHOD_REQUEST_LOAD)
            } catch (e: RemoteException) {
                Log.i(TAG, "Provider ${contentUri.authority} crashed while requesting load", e)
            }
        }

        suspend fun getDescription(context: Context, authority: String): String {
            val contentUri = Uri.Builder()
                    .scheme(ContentResolver.SCHEME_CONTENT)
                    .authority(authority)
                    .build()
            return ContentProviderClientCompat.getClient(context, contentUri)?.use { client ->
                try {
                    val result = client.call(ProtocolConstants.METHOD_GET_DESCRIPTION)
                    result?.getString(ProtocolConstants.KEY_DESCRIPTION, "") ?: ""
                } catch (e: RemoteException) {
                    Log.i(TAG, "Provider $authority crashed while retrieving description", e)
                    ""
                }
            } ?: ""
        }
    }
    private val packageChangeReceiver : BroadcastReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent?) {
            val packageName = intent?.data?.schemeSpecificPart
            val changedComponents = intent?.getStringArrayExtra(
                    Intent.EXTRA_CHANGED_COMPONENT_NAME_LIST) ?: emptyArray()
            val pm = context.packageManager
            // Restart loading if any active provider (home and/or lock) was affected
            val affected = activeProviders.distinctBy { it.authority }.any { provider ->
                @Suppress("DEPRECATION")
                @SuppressLint("InlinedApi")
                val providerInfo = pm.resolveContentProvider(provider.authority,
                        PackageManager.MATCH_DISABLED_COMPONENTS)
                val providerComponentName = providerInfo?.name
                val wholePackageChanged = changedComponents.any { it == packageName }
                val providerChanged = providerInfo != null
                        && changedComponents.any { it == providerComponentName }
                providerInfo == null || (providerInfo.packageName == packageName
                        && (wholePackageChanged || providerChanged))
            }
            if (affected) {
                // A selected provider changed, so restart loading
                startArtworkLoad()
            }
        }
    }
    private val contentObserver = object : ContentObserver(Handler(Looper.getMainLooper())) {
        override fun onChange(selfChange: Boolean, uri: Uri?) {
            if (BuildConfig.DEBUG) {
                Log.d(TAG, "onChange for $uri")
            }
            ProviderChangedWorker.enqueueChanged(context)
        }
    }
    private val providerLiveData by lazy {
        MuzeiDatabase.getInstance(context).providerDao().getCurrentProviderLiveData()
    }
    private val allProvidersLiveData by lazy {
        MuzeiDatabase.getInstance(context).providerDao().getAllProvidersLiveData()
    }
    private val artworkLiveData by lazy {
        MuzeiDatabase.getInstance(context).artworkDao().getCurrentArtworkLiveData()
    }
    /**
     * The set of currently selected providers (home and, when unlinked, lock),
     * cached so the package-change receiver and artwork load can react to all of
     * them without a database round-trip.
     */
    private var activeProviders: List<Provider> = emptyList()
    private val allProvidersObserver = Observer<List<Provider>> { providers ->
        val previous = activeProviders.map { it.authority }.toSet()
        val current = providers.map { it.authority }.toSet()
        activeProviders = providers
        if (current != previous) {
            // The set of selected providers changed (home or lock), restart loading
            startArtworkLoad()
        }
    }
    private var nextArtworkJob: Job? = null
    @OptIn(DelicateCoroutinesApi::class)
    private val artworkObserver = Observer<Artwork?> { artwork ->
        if (artwork == null) {
            // Can't have no artwork at all,
            // try loading the next artwork with a slight delay
            nextArtworkJob?.cancel()
            nextArtworkJob = GlobalScope.launch {
                delay(1000)
                if (nextArtworkJob?.isCancelled == false) {
                    nextArtwork()
                }
            }
        } else {
            nextArtworkJob?.cancel()
        }
    }

    var loadFrequencySeconds: Long
        set(newLoadFrequency) {
            PreferenceManager.getDefaultSharedPreferences(context).edit {
                putLong(PREF_LOAD_FREQUENCY_SECONDS, newLoadFrequency)
            }
            if (newLoadFrequency > 0) {
                ArtworkLoadWorker.enqueuePeriodic(context, newLoadFrequency, loadOnWifi)
            } else {
                ArtworkLoadWorker.cancelPeriodic(context)
            }
        }
        get() = PreferenceManager.getDefaultSharedPreferences(context)
                .getLong(PREF_LOAD_FREQUENCY_SECONDS, DEFAULT_LOAD_FREQUENCY_SECONDS)

    var loadOnWifi: Boolean
        set(newLoadOnWifi) {
            PreferenceManager.getDefaultSharedPreferences(context).edit {
                putBoolean(PREF_LOAD_ON_WIFI, newLoadOnWifi)
            }
            if (loadFrequencySeconds > 0) {
                ArtworkLoadWorker.enqueuePeriodic(context, loadFrequencySeconds, newLoadOnWifi)
            }
        }
        get() = PreferenceManager.getDefaultSharedPreferences(context)
                .getBoolean(PREF_LOAD_ON_WIFI, DEFAULT_LOAD_ON_WIFI)

    var loadOrdering: LoadOrdering
        set(newLoadOrdering) {
            PreferenceManager.getDefaultSharedPreferences(context).edit {
                putString(PREF_LOAD_ORDERING, newLoadOrdering.name)
            }
        }
        get() = LoadOrdering.valueOf(checkNotNull(PreferenceManager.getDefaultSharedPreferences(context)
            .getString(PREF_LOAD_ORDERING, DEFAULT_LOAD_ORDERING.name)) {
            "Invalid load ordering"
        })

    @SuppressLint("WrongConstant")
    override fun onActive() {
        if (BuildConfig.DEBUG) {
            Log.d(TAG, "ProviderManager became active")
        }
        // Register for package change events
        val packageChangeFilter = IntentFilter().apply {
            addDataScheme("package")
            addAction(Intent.ACTION_PACKAGE_ADDED)
            addAction(Intent.ACTION_PACKAGE_CHANGED)
            addAction(Intent.ACTION_PACKAGE_REPLACED)
            addAction(Intent.ACTION_PACKAGE_REMOVED)
        }
        ContextCompat.registerReceiver(
            context,
            packageChangeReceiver,
            packageChangeFilter,
            ContextCompat.RECEIVER_NOT_EXPORTED
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            ProviderChangedWorker.activeListeningStateChanged(context, true)
        }
        providerLiveData.observeForever(this)
        allProvidersLiveData.observeForever(allProvidersObserver)
        artworkLiveData.observeForever(artworkObserver)
        // allProvidersObserver kicks off the initial load once the provider set
        // is delivered; an empty set means there is nothing to load yet.
    }

    @OptIn(DelicateCoroutinesApi::class)
    private fun runIfValid(provider: Provider?, block: (provider: Provider) -> Unit) {
        if (provider != null) {
            val pm = context.packageManager
            @Suppress("DEPRECATION")
            if (pm.resolveContentProvider(provider.authority, 0) != null) {
                // resolveContentProvider succeeded, so it is a valid ContentProvider
                block(provider)
            } else {
                // Invalid ContentProvider, remove it from the ProviderDao
                GlobalScope.launch {
                    if (BuildConfig.DEBUG) {
                        Log.w(TAG, "Invalid provider ${provider.authority}")
                    }
                    MuzeiDatabase.getInstance(context).providerDao().delete(provider)
                }
            }
        }
    }

    private fun startArtworkLoad() {
        if (!hasActiveObservers()) {
            return
        }
        if (BuildConfig.DEBUG) {
            Log.d(TAG, "Starting artwork load")
        }
        // Re-register the content observer for every active provider (home and,
        // when unlinked, lock). Unregister first so we don't accumulate stale
        // registrations for providers that are no longer selected.
        context.contentResolver.unregisterContentObserver(contentObserver)
        activeProviders.distinctBy { it.authority }.forEach { provider ->
            runIfValid(provider) { valid ->
                // Listen for MuzeiArtProvider changes
                val contentUri = ProviderContract.getContentUri(valid.authority)
                context.contentResolver.registerContentObserver(
                        contentUri, true, contentObserver)
            }
        }
        ProviderChangedWorker.enqueueSelected(context)
    }

    override fun onChanged(value: Provider?) {
        // Track the home provider so consumers observing ProviderManager (e.g. the
        // null -> default-provider fallback) see the home screen's selection.
        // Loading is driven by allProvidersObserver, which covers both screens.
        this.value = value
        // Drop the home provider if it is no longer a valid ContentProvider
        runIfValid(value) { }
    }

    override fun onInactive() {
        nextArtworkJob?.cancel()
        artworkLiveData.removeObserver(artworkObserver)
        allProvidersLiveData.removeObserver(allProvidersObserver)
        // Clear the cache so the next activation observes an empty -> selected
        // transition and kicks off a fresh load.
        activeProviders = emptyList()
        providerLiveData.removeObserver(this)
        context.contentResolver.unregisterContentObserver(contentObserver)
        ArtworkLoadWorker.cancelPeriodic(context)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            ProviderChangedWorker.activeListeningStateChanged(context, false)
        }
        context.unregisterReceiver(packageChangeReceiver)
        if (BuildConfig.DEBUG) {
            Log.d(TAG, "ProviderManager is now inactive")
        }
    }

    @OptIn(DelicateCoroutinesApi::class)
    fun nextArtwork(screen: Screen = Screen.HOME) {
        GlobalScope.launch {
            val providerDao = MuzeiDatabase.getInstance(context).providerDao()
            // When the lock screen is linked (has no provider of its own) it shows the
            // home artwork, so advance the home screen instead.
            val targetScreen = if (screen == Screen.LOCK &&
                    providerDao.getProvider(Screen.LOCK.value) == null) {
                Screen.HOME
            } else {
                screen
            }
            ArtworkLoadWorker.enqueueNext(context, targetScreen)
        }
    }
}