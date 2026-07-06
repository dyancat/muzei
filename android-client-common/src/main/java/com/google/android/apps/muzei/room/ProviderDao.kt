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

package com.google.android.apps.muzei.room

import androidx.lifecycle.LiveData
import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Transaction
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

/**
 * Dao for Providers
 */
@Dao
abstract class ProviderDao {
    // The no-arg queries below describe the HOME screen (screen = 0). They retain
    // their original "the current provider" meaning so the many existing callers
    // and the public MuzeiContract ContentProvider stay correct and single-row.
    @Query("SELECT * FROM provider WHERE screen = 0")
    abstract fun getCurrentProviderFlow(): Flow<Provider?>

    @Query("SELECT * FROM provider WHERE screen = 0")
    abstract fun getCurrentProviderLiveData(): LiveData<Provider?>

    @Query("SELECT * FROM provider WHERE screen = 0")
    internal abstract fun getCurrentProviderBlocking(): Provider?

    @Query("SELECT * FROM provider WHERE screen = 0")
    abstract suspend fun getCurrentProvider(): Provider?

    @Query("SELECT * FROM provider WHERE screen = :screen")
    abstract fun getProviderFlow(screen: Int): Flow<Provider?>

    @Query("SELECT * FROM provider WHERE screen = :screen")
    abstract suspend fun getProvider(screen: Int): Provider?

    @Query("SELECT * FROM provider")
    abstract fun getAllProvidersFlow(): Flow<List<Provider>>

    @Query("SELECT * FROM provider")
    abstract fun getAllProvidersLiveData(): LiveData<List<Provider>>

    @Query("SELECT * FROM provider")
    abstract suspend fun getAllProviders(): List<Provider>

    @Transaction
    open suspend fun select(authority: String, screen: Int = Screen.HOME.value) {
        deleteByScreen(screen)
        insert(Provider(screen, authority))
    }

    @Insert
    internal abstract suspend fun insert(provider: Provider)

    @Update
    abstract suspend fun update(provider: Provider)

    @Delete
    abstract suspend fun delete(provider: Provider)

    @Query("DELETE FROM provider WHERE screen = :screen")
    internal abstract suspend fun deleteByScreen(screen: Int)

    /**
     * Remove the lock screen's provider selection, "linking" the lock screen back
     * to the home screen's provider.
     */
    @Query("DELETE FROM provider WHERE screen = 1")
    abstract suspend fun clearLock()

    @Query("DELETE FROM provider")
    internal abstract suspend fun deleteAll()
}
