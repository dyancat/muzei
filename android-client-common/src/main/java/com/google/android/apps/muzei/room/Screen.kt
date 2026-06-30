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

package com.google.android.apps.muzei.room

/**
 * The screen a provider selection applies to. Persisted as [value] in the
 * provider table's primary key so the home and lock screens can each use a
 * different provider.
 *
 * The absence of a [LOCK] row means the lock screen is "linked" to the home
 * screen and falls back to the [HOME] selection.
 */
enum class Screen(val value: Int) {
    HOME(0),
    LOCK(1);

    companion object {
        fun fromValue(value: Int): Screen = entries.first { it.value == value }
    }
}
