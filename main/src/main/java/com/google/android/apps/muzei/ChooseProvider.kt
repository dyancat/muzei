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

package com.google.android.apps.muzei

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.PagerState
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Link
import androidx.compose.material.icons.filled.LinkOff
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Update
import androidx.compose.material3.DrawerDefaults
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SecondaryTabRow
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Devices.PHONE
import androidx.compose.ui.tooling.preview.Devices.TABLET
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.graphics.drawable.toDrawable
import androidx.core.net.toUri
import coil3.ColorImage
import coil3.annotation.ExperimentalCoilApi
import coil3.compose.AsyncImagePreviewHandler
import coil3.compose.LocalAsyncImagePreviewHandler
import com.google.android.apps.muzei.theme.AppTheme
import com.google.android.apps.muzei.util.ModalRightNavigationDrawer
import com.google.android.apps.muzei.util.plus
import com.google.firebase.Firebase
import com.google.firebase.analytics.analytics
import kotlinx.coroutines.launch
import net.nurik.roman.muzei.R

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun ChooseProvider(
    homeProviders: List<ProviderInfo>,
    lockProviders: List<ProviderInfo>,
    modifier: Modifier = Modifier,
    pagerState: PagerState = rememberPagerState(initialPage = 0, pageCount = { 2 }),
    lockLinked: Boolean = true,
    onToggleLockLink: () -> Unit = {},
    drawerSheetContent: @Composable ColumnScope.() -> Unit = {},
    drawerSheetContainerColor: Color = DrawerDefaults.modalContainerColor,
    onNotificationSettingsClick: () -> Unit = {},
    autoScrollToProviderAuthority: String? = null,
    onAutoScrollToProviderCompleted: () -> Unit = {},
    onClick: (ProviderInfo) -> Unit = {},
    onLongClick: (ProviderInfo) -> Unit = {},
    onSettingsClick: (ProviderInfo) -> Unit = {},
    onBrowseClick: (ProviderInfo) -> Unit = {},
) {
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    ModalRightNavigationDrawer(
        drawerSheetContent = drawerSheetContent,
        modifier = modifier,
        drawerState = drawerState,
        drawerSheetContainerColor = drawerSheetContainerColor,
        // The Home/Lock tabs are a horizontal pager; disable the drawer's swipe-to-open
        // so it doesn't fight that gesture. The Auto Advance drawer still opens from the
        // toolbar's Update button (and closes via the scrim or back).
        gesturesEnabled = false,
    ) {
        Scaffold(
            topBar = {
                // A light scrim behind the whole top area (app bar + tabs, including
                // the status bar inset) so they stay legible over bright artwork.
                Column(
                    modifier = Modifier.background(Color.Black.copy(alpha = 0.3f)),
                ) {
                    TopAppBar(
                        title = {},
                        actions = {
                            // On the lock screen tab, offer linking the lock screen
                            // back to the home screen's provider (no separate source).
                            if (pagerState.currentPage == 1) {
                                IconButton(onClick = onToggleLockLink) {
                                    Icon(
                                        if (lockLinked) Icons.Default.Link
                                        else Icons.Default.LinkOff,
                                        contentDescription = stringResource(
                                            if (lockLinked) R.string.action_link_lock_source
                                            else R.string.action_link_lock_source_off
                                        ),
                                    )
                                }
                            }
                            val coroutineScope = rememberCoroutineScope()
                            IconButton(
                                onClick = {
                                    coroutineScope.launch {
                                        Firebase.analytics.logEvent("auto_advance_open", null)
                                        drawerState.open()
                                    }
                                },
                            ) {
                                Icon(
                                    Icons.Default.Update,
                                    contentDescription = stringResource(R.string.auto_advance_settings),
                                )
                            }
                            // Show the menu items in a DropdownMenu
                            var expanded by remember { mutableStateOf(false) }
                            IconButton(onClick = { expanded = !expanded }) {
                                Icon(
                                    Icons.Default.MoreVert,
                                    contentDescription = null
                                )
                            }
                            DropdownMenu(
                                expanded = expanded,
                                onDismissRequest = { expanded = false }
                            ) {
                                DropdownMenuItem(
                                    text = { Text(stringResource(R.string.notification_settings)) },
                                    onClick = {
                                        expanded = false
                                        onNotificationSettingsClick()
                                    }
                                )
                            }
                        },
                        colors = TopAppBarDefaults.topAppBarColors(
                            containerColor = Color.Transparent,
                            scrolledContainerColor = Color.Transparent,
                            navigationIconContentColor = Color.White,
                            titleContentColor = Color.White,
                            actionIconContentColor = Color.White,
                            subtitleContentColor = Color.White,
                        ),
                    )
                    SecondaryTabRow(
                        selectedTabIndex = pagerState.currentPage,
                        containerColor = Color.Transparent,
                        contentColor = Color.White,
                    ) {
                        val tabCoroutineScope = rememberCoroutineScope()
                        val tabs = listOf(
                            stringResource(R.string.settings_home_screen_title),
                            stringResource(R.string.settings_lock_screen_title),
                        )
                        tabs.forEachIndexed { index, tabTitle ->
                            Tab(
                                selected = pagerState.currentPage == index,
                                onClick = {
                                    tabCoroutineScope.launch {
                                        pagerState.animateScrollToPage(index)
                                    }
                                },
                                text = { Text(text = tabTitle) },
                            )
                        }
                    }
                }
            },
            containerColor = Color.Transparent,
            contentColor = Color.White,
        ) { innerPadding ->
            // A page per screen (home/lock) so the user can swipe between the tabs.
            // Each page renders its own screen's list, so a page's checkmark and
            // artwork are fixed to that screen and don't change (or animate) when
            // the other tab is selected.
            //
            // Keep the off-screen page composed (rather than the default of
            // disposing it once settled). Each page is a full staggered grid of
            // fairly heavy cards, and recomposing/measuring it from scratch on
            // every swipe janks the slide animation.
            HorizontalPager(
                state = pagerState,
                beyondViewportPageCount = 1,
            ) { page ->
                val pageProviders = if (page == 0) homeProviders else lockProviders
                // While the lock screen is linked to home its provider can't be
                // chosen separately, so its page is disabled until it's unlinked.
                val pageEnabled = page == 0 || !lockLinked
                // A non-lazy, scrolling flow rather than a LazyVerticalStaggeredGrid:
                // a lazy grid nested in the pager composes items only while its own
                // viewport is on-screen, so it disposes and recomposes a whole
                // screenful of (fairly heavy) cards on every tab switch, janking the
                // slide. Provider lists are short, so composing all cards up front
                // and keeping both pages resident makes a switch a pure translation.
                val scrollState = rememberScrollState()
                BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
                    val spacing = 16.dp
                    val contentWidth = maxWidth - spacing * 2
                    val columns = maxOf(1, ((contentWidth + spacing) / (300.dp + spacing)).toInt())
                    val cardWidth = (contentWidth - spacing * (columns - 1)) / columns
                    FlowRow(
                        modifier = Modifier
                            .fillMaxWidth()
                            .verticalScroll(scrollState)
                            // Padding lives inside the scroll so the top/bottom
                            // breathing room scrolls with the content, matching the
                            // grid's old contentPadding.
                            .padding(innerPadding + PaddingValues(all = spacing)),
                        horizontalArrangement = Arrangement.spacedBy(spacing),
                        verticalArrangement = Arrangement.spacedBy(spacing),
                        maxItemsInEachRow = columns,
                    ) {
                        pageProviders.forEach { providerInfo ->
                            val bringIntoView = remember { BringIntoViewRequester() }
                            val isAutoScrollTarget =
                                providerInfo.authority == autoScrollToProviderAuthority
                            LaunchedEffect(isAutoScrollTarget) {
                                if (isAutoScrollTarget) {
                                    bringIntoView.bringIntoView()
                                    onAutoScrollToProviderCompleted()
                                }
                            }
                            ChooseProviderItem(
                                providerInfo = providerInfo,
                                modifier = Modifier
                                    .width(cardWidth)
                                    .bringIntoViewRequester(bringIntoView),
                                enabled = pageEnabled,
                                onClick = {
                                    onClick(providerInfo)
                                },
                                onLongClick = {
                                    onLongClick(providerInfo)
                                },
                                onSettingsClick = {
                                    onSettingsClick(providerInfo)
                                },
                                onBrowseClick = {
                                    onBrowseClick(providerInfo)
                                }
                            )
                        }
                    }
                }
            }
        }
    }
}

@Preview(name = "Portrait", device = PHONE)
@Preview(
    name = "Landscape",
    device = "spec:width=411dp,height=891dp,orientation=landscape,dpi=420",
)
@Preview(name = "Tablet - Landscape", device = TABLET)
@OptIn(ExperimentalCoilApi::class)
@Composable
fun ChooseProviderPreview() {
    val providers = remember {
        mutableStateListOf(
            // Default
            ProviderInfo(
                authority = "muzei:default",
                packageName = "",
                title = "Muzei",
                description = "A new painting every day",
                currentArtworkUri = "0".toUri(),
                icon = Color.Magenta.toArgb().toDrawable(),
                setupActivity = null,
                settingsActivity = null,
                selected = false
            ),
            // No description
            ProviderInfo(
                authority = "muzei:no_description",
                packageName = "",
                title = "Muzei",
                description = null,
                currentArtworkUri = "0".toUri(),
                icon = Color.Magenta.toArgb().toDrawable(),
                setupActivity = null,
                settingsActivity = null,
                selected = false
            ),
            // No currentArtworkUri
            ProviderInfo(
                authority = "muzei:no_artwork",
                packageName = "",
                title = "Muzei",
                description = "A new painting every day",
                currentArtworkUri = null,
                icon = Color.Magenta.toArgb().toDrawable(),
                setupActivity = null,
                settingsActivity = null,
                selected = false
            ),
            // No description or currentArtworkUri
            ProviderInfo(
                authority = "muzei_empty",
                packageName = "",
                title = "Muzei",
                description = null,
                currentArtworkUri = null,
                icon = Color.Magenta.toArgb().toDrawable(),
                setupActivity = null,
                settingsActivity = null,
                selected = false
            ),
        )
    }
    val colors = listOf(Color.Red, Color.Green, Color.Blue, Color.Yellow)
    val previewHandler = AsyncImagePreviewHandler { request ->
        ColorImage(colors[request.data.toString().toInt().mod(colors.size)].toArgb())
    }
    CompositionLocalProvider(LocalAsyncImagePreviewHandler provides previewHandler) {
        AppTheme(
            dynamicColor = false
        ) {
            ChooseProvider(
                homeProviders = providers,
                lockProviders = providers,
            )
        }
    }
}
