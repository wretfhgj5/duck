'use strict';

/*
 * ============================================================
 * Duck.ai -> OpenAI Compatible API
 * Railway + Concurrent Edition
 *
 * Endpoints:
 *   GET  /
 *   GET  /health
 *   GET  /v1/models
 *   POST /v1/chat/completions
 *   POST /v1/responses
 *
 * Compatibility:
 *   GET  /models
 *   POST /chat/completions
 *   POST /api/chat
 *
 * Railway:
 *   - PORT comes from Railway
 *   - Listen on 0.0.0.0:$PORT
 *   - Chromium path comes from CHROME_PATH
 *   - MAX_CONCURRENCY defaults to 2
 * ============================================================
 */

const express = require('express');
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// ============================================================
// APP
// ============================================================

const app = express();

app.disable('x-powered-by');

app.use(
    express.json({
        limit: '8mb',
        strict: false
    })
);

// ============================================================
// CONFIG
// ============================================================

const PORT = Number(
    process.env.PORT || 3000
);

const HOST =
    process.env.HOST || '0.0.0.0';

const DUCK_URL =
    process.env.DUCK_URL || 'https://duck.ai';

const PAGE_TIMEOUT = Number(
    process.env.PAGE_TIMEOUT || 60000
);

const RESPONSE_TIMEOUT = Number(
    process.env.RESPONSE_TIMEOUT || 80000
);

const POLL_INTERVAL = Number(
    process.env.POLL_INTERVAL || 350
);

const STABLE_POLLS = Number(
    process.env.STABLE_POLLS || 6
);

const MAX_RETRIES = Number(
    process.env.MAX_RETRIES || 1
);

const MAX_CONCURRENCY = Math.max(
    1,
    Number(
        process.env.MAX_CONCURRENCY || 2
    )
);

const STREAM_DELAY = Number(
    process.env.STREAM_DELAY || 8
);

const KEEP_ALIVE_INTERVAL = Number(
    process.env.KEEP_ALIVE_INTERVAL || 0
);

const DEBUG_ENABLED =
    String(
        process.env.DEBUG || 'false'
    ).toLowerCase() === 'true';

const REQUIRE_API_KEY =
    String(
        process.env.REQUIRE_API_KEY || 'false'
    ).toLowerCase() === 'true';

const API_KEY =
    process.env.API_KEY || '';

const CHROME_PATH =
    process.env.CHROME_PATH || '';

const DEBUG_DIR =
    path.join(
        __dirname,
        'duck-debug'
    );

if (!fs.existsSync(DEBUG_DIR)) {
    fs.mkdirSync(
        DEBUG_DIR,
        {
            recursive: true
        }
    );
}

// ============================================================
// MODELS
// ============================================================

const MODELS = {

    'gpt-5.4-nano': {
        id: 'gpt-5.4-nano',

        labels: [
            'GPT-5.4 nano',
            '5.4-nano'
        ],

        aliases: [
            'gpt-5.4-nano',
            '5.4-nano',
            'GPT-5.4 nano',
            'GPT-5.4 Nano'
        ]
    },

    'gpt-5.4-mini': {
        id: 'gpt-5.4-mini',

        labels: [
            'GPT-5.4 mini',
            '5.4-mini'
        ],

        aliases: [
            'gpt-5.4-mini',
            '5.4-mini',
            'GPT-5.4 mini',
            'GPT-5.4 Mini'
        ]
    },

    'claude-4.5-haiku': {
        id: 'claude-4.5-haiku',

        labels: [
            'Claude Haiku 4.5',
            'Claude 4.5 Haiku',
            'Haiku 4.5',
            'Claude Haiku'
        ],

        aliases: [
            'claude-4.5-haiku',
            'Claude Haiku 4.5',
            'Claude 4.5 Haiku',
            'Haiku 4.5',
            'Claude Haiku'
        ]
    },

    'mistral-small-4': {
        id: 'mistral-small-4',

        labels: [
            'Mistral Small 4',
            'Mistral Small',
            'Mistral'
        ],

        aliases: [
            'mistral-small-4',
            'Mistral Small 4',
            'Mistral Small',
            'Mistral'
        ]
    },

    'gpt-oss-120b': {
        id: 'gpt-oss-120b',

        labels: [
            'gpt-oss 120B',
            'GPT-OSS 120B',
            'gpt-oss'
        ],

        aliases: [
            'gpt-oss-120b',
            'gpt-oss 120B',
            'GPT-OSS 120B',
            'gpt-oss'
        ]
    },

    'gemma-4-31b': {
        id: 'gemma-4-31b',

        labels: [
            'Gemma 4 31B',
            'Gemma 4',
            'Gemma'
        ],

        aliases: [
            'gemma-4-31b',
            'Gemma 4 31B',
            'Gemma 4',
            'Gemma'
        ]
    }
};

const COMPAT_ALIASES = {
    'gpt-4o-mini': 'gpt-5.4-nano',
    'gpt-4o': 'gpt-5.4-mini',
    'o3-mini': 'gpt-5.4-mini',
    'claude-haiku': 'claude-4.5-haiku'
};

const DEFAULT_MODEL =
    'gpt-5.4-nano';

// ============================================================
// GLOBAL STATE
// ============================================================

let browser = null;
let browserStarting = null;

let shuttingDown = false;
let requestCounter = 0;

// ============================================================
// CONCURRENCY POOL
// ============================================================

class ConcurrencyPool {

    constructor(size) {
        this.size = Math.max(
            1,
            Number(size)
        );

        this.active = 0;
        this.waiting = [];
        this.completedCount = 0;
        this.failedCount = 0;
    }

    get available() {
        return Math.max(
            0,
            this.size - this.active
        );
    }

    async acquire() {

        if (
            this.active <
            this.size
        ) {
            this.active++;

            if (DEBUG_ENABLED) {
                console.log(
                    `🟢 Pool slot acquired: ${this.active}/${this.size}`
                );
            }

            return;
        }

        if (DEBUG_ENABLED) {
            console.log(
                `⏳ Pool full: active=${this.active}/${this.size}, waiting=${this.waiting.length + 1}`
            );
        }

        await new Promise(
            resolve => {
                this.waiting.push(
                    resolve
                );
            }
        );
    }

    release() {

        if (
            this.waiting.length > 0
        ) {

            const next =
                this.waiting.shift();

            if (DEBUG_ENABLED) {
                console.log(
                    `🔄 Pool slot passed to waiting request`
                );
            }

            next();
            return;
        }

        this.active =
            Math.max(
                0,
                this.active - 1
            );

        if (DEBUG_ENABLED) {
            console.log(
                `🔵 Pool slot released: ${this.active}/${this.size}`
            );
        }
    }

    completed() {
        this.completedCount++;
    }

    failed() {
        this.failedCount++;
    }
}

const pool =
    new ConcurrencyPool(
        MAX_CONCURRENCY
    );

// ============================================================
// HELPERS
// ============================================================

function sleep(ms) {
    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                ms
            )
    );
}

function cleanText(text) {
    return String(text || '')
        .replace(/\r/g, '')
        .replace(/\u00a0/g, ' ')
        .replace(/[ \t]+\n/g, '\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
}

function normalize(text) {
    return cleanText(text)
        .replace(/\s+/g, ' ')
        .trim()
        .toLowerCase();
}

function timestamp() {
    return new Date()
        .toISOString()
        .replace(
            /[:.]/g,
            '-'
        );
}

function makeId(prefix) {
    return (
        `${prefix}-${Date.now()}-` +
        crypto
            .randomBytes(5)
            .toString('hex')
    );
}

// ============================================================
// MODEL
// ============================================================

function resolveModel(requested) {
    let raw = String(requested || '').trim();

    // Accept "duck/gpt-5.4-nano" or any provider prefix
    const afterLastSlash = raw.includes('/')
        ? raw.split('/').pop().trim()
        : raw;

    const candidates = [raw, afterLastSlash];

    for (const candidate of candidates) {
        if (MODELS[candidate]) {
            return {
                requested: raw,
                actual: candidate,
                aliased: candidate !== raw
            };
        }

        if (COMPAT_ALIASES[candidate]) {
            return {
                requested: raw,
                actual: COMPAT_ALIASES[candidate],
                aliased: true
            };
        }
    }

    const lower = raw.toLowerCase();
    const lowerAfterSlash = afterLastSlash.toLowerCase();

    for (const [id, definition] of Object.entries(MODELS)) {
        const values = [
            id,
            ...(definition.labels || []),
            ...(definition.aliases || [])
        ];

        if (values.some(value => {
            const v = String(value).toLowerCase();
            return v === lower || v === lowerAfterSlash;
        })) {
            return {
                requested: raw,
                actual: id,
                aliased: lowerAfterSlash !== id.toLowerCase()
            };
        }
    }

    throw Object.assign(
        new Error(
            `The model '${raw}' does not exist or is not available through this proxy.`
        ),
        {
            status: 400,
            code: 'model_not_found'
        }
    );
}

// ============================================================
// CONTENT
// ============================================================

function contentToText(content) {

    if (
        typeof content ===
        'string'
    ) {
        return content;
    }

    if (
        content == null
    ) {
        return '';
    }

    if (
        Array.isArray(content)
    ) {
        return content
            .map(
                part => {

                    if (
                        typeof part ===
                        'string'
                    ) {
                        return part;
                    }

                    if (
                        part &&
                        (
                            part.type === 'text' ||
                            part.type === 'input_text'
                        ) &&
                        typeof part.text ===
                            'string'
                    ) {
                        return part.text;
                    }

                    return '';
                }
            )
            .filter(Boolean)
            .join('\n');
    }

    return String(content);
}

function validateMessages(
    messages
) {

    if (
        !Array.isArray(messages) ||
        messages.length === 0
    ) {
        throw Object.assign(
            new Error(
                'messages must be a non-empty array'
            ),
            {
                status: 400,
                code: 'invalid_messages'
            }
        );
    }

    for (
        const message of messages
    ) {

        if (
            !message ||
            typeof message !== 'object'
        ) {
            throw Object.assign(
                new Error(
                    'Each message must be an object'
                ),
                {
                    status: 400,
                    code: 'invalid_message'
                }
            );
        }

        if (
            ![
                'system',
                'developer',
                'user',
                'assistant'
            ].includes(
                message.role
            )
        ) {
            throw Object.assign(
                new Error(
                    `Unsupported message role: ${message.role}`
                ),
                {
                    status: 400,
                    code: 'invalid_role'
                }
            );
        }

        if (
            !contentToText(
                message.content
            ).trim()
        ) {
            throw Object.assign(
                new Error(
                    'Message content cannot be empty'
                ),
                {
                    status: 400,
                    code: 'empty_content'
                }
            );
        }
    }
}

function getLastUserMessage(
    messages
) {

    for (
        let i = messages.length - 1;
        i >= 0;
        i--
    ) {

        if (
            messages[i] &&
            messages[i].role === 'user'
        ) {
            return contentToText(
                messages[i].content
            ).trim();
        }
    }

    return '';
}

// ============================================================
// BUILD DUCK PROMPT
// ============================================================

function buildDuckPrompt(
    messages
) {

    validateMessages(
        messages
    );

    if (
        messages.length === 1 &&
        messages[0].role === 'user'
    ) {
        return contentToText(
            messages[0].content
        ).trim();
    }

    const systemParts = [];
    const conversation = [];

    for (
        const message of messages
    ) {

        const text =
            contentToText(
                message.content
            ).trim();

        if (!text) {
            continue;
        }

        if (
            message.role === 'system' ||
            message.role === 'developer'
        ) {
            systemParts.push(text);
            continue;
        }

        conversation.push({
            role: message.role,
            content: text
        });
    }

    const blocks = [];

    if (
        systemParts.length > 0
    ) {
        blocks.push(
            [
                'System instructions:',
                systemParts.join(
                    '\n\n'
                )
            ].join('\n')
        );
    }

    if (
        conversation.length > 0
    ) {
        blocks.push(
            conversation
                .map(item => {

                    const label =
                        item.role === 'assistant'
                            ? 'Assistant'
                            : 'User';

                    return (
                        `${label}:\n` +
                        item.content
                    );
                })
                .join(
                    '\n\n'
                )
        );
    }

    blocks.push(
        'Answer the latest user message directly.'
    );

    blocks.push(
        `Latest user message:\n${getLastUserMessage(
            messages
        )}`
    );

    return blocks
        .join('\n\n')
        .trim();
}

// ============================================================
// BROWSER PATH
// ============================================================

function findBrowser() {

    const linuxCandidates = [
        CHROME_PATH,
        '/usr/bin/chromium',
        '/usr/bin/chromium-browser',
        '/usr/bin/google-chrome',
        '/usr/bin/google-chrome-stable'
    ];

    const windowsCandidates = [
        process.env.CHROME_PATH,

        'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',

        'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',

        path.join(
            process.env.LOCALAPPDATA || '',
            'Google\\Chrome\\Application\\chrome.exe'
        ),

        'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',

        'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
    ];

    const candidates =
        process.platform === 'win32'
            ? [
                ...windowsCandidates,
                ...linuxCandidates
            ]
            : [
                ...linuxCandidates,
                ...windowsCandidates
            ];

    for (
        const candidate of candidates
    ) {

        if (!candidate) {
            continue;
        }

        try {

            if (
                fs.existsSync(
                    candidate
                )
            ) {

                console.log(
                    `✅ Browser found: ${candidate}`
                );

                return candidate;
            }

        } catch (_) {}
    }

    throw new Error(
        'Chromium/Chrome not found. Set CHROME_PATH.'
    );
}

// ============================================================
// BROWSER
// ============================================================

async function getBrowser() {

    if (
        browser &&
        browser.isConnected()
    ) {
        return browser;
    }

    if (browserStarting) {
        return browserStarting;
    }

    browserStarting =
        (async () => {

            const executablePath =
                findBrowser();

            console.log(
                '🚀 Starting browser...'
            );

            const instance =
                await puppeteer.launch({
                    executablePath,

                    headless: true,

                    defaultViewport: {
                        width: 1920,
                        height: 1080
                    },

                    args: [
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--disable-notifications',
                        '--disable-popup-blocking',
                        '--disable-background-networking',
                        '--disable-background-timer-throttling',
                        '--disable-renderer-backgrounding',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-features=Translate,BackForwardCache',
                        '--window-size=1920,1080'
                    ]
                });

            instance.on(
                'disconnected',
                () => {
                    browser = null;
                }
            );

            browser =
                instance;

            console.log(
                '✅ Browser started'
            );

            return instance;
        })();

    try {
        return await browserStarting;
    } finally {
        browserStarting =
            null;
    }
}

// ============================================================
// PAGE SETUP
// ============================================================

async function setupPage(page) {

    await page.setUserAgent(
        'Mozilla/5.0 (X11; Linux x86_64) ' +
        'AppleWebKit/537.36 (KHTML, like Gecko) ' +
        'Chrome/151.0.0.0 Safari/537.36'
    );

    await page.setViewport({
        width: 1920,
        height: 1080
    });

    page.setDefaultTimeout(
        20000
    );

    page.setDefaultNavigationTimeout(
        PAGE_TIMEOUT
    );

    await page.evaluateOnNewDocument(
        () => {

            try {

                Object.defineProperty(
                    navigator,
                    'webdriver',
                    {
                        configurable: true,
                        get: () =>
                            undefined
                    }
                );

            } catch (_) {}

            try {

                Object.defineProperty(
                    navigator,
                    'languages',
                    {
                        configurable: true,
                        get: () => [
                            'en-US',
                            'en'
                        ]
                    }
                );

            } catch (_) {}

            try {

                Object.defineProperty(
                    navigator,
                    'plugins',
                    {
                        configurable: true,
                        get: () => [
                            1, 2, 3, 4, 5
                        ]
                    }
                );

            } catch (_) {}
        }
    );

    page.on(
        'pageerror',
        error => {
            if (DEBUG_ENABLED) {
                console.log(
                    `🌐 PAGE ERROR: ${error.message}`
                );
            }
        }
    );

    page.on(
        'requestfailed',
        request => {

            const failure =
                request.failure();

            if (
                failure &&
                !String(
                    failure.errorText || ''
                ).includes(
                    'ERR_ABORTED'
                )
            ) {

                if (DEBUG_ENABLED) {
                    console.log(
                        `🌐 REQUEST FAILED: ${request.url()} -> ${failure.errorText}`
                    );
                }
            }
        }
    );
}

// ============================================================
// DEBUG
// ============================================================

async function saveDebug(
    page,
    prefix
) {

    if (
        !DEBUG_ENABLED ||
        !page ||
        page.isClosed()
    ) {
        return;
    }

    const stamp =
        timestamp();

    try {

        const screenshot =
            path.join(
                DEBUG_DIR,
                `${prefix}-${stamp}.png`
            );

        await page.screenshot({
            path: screenshot,
            fullPage: true
        });

        console.log(
            `📸 ${screenshot}`
        );

    } catch (error) {

        console.log(
            `⚠️ Screenshot failed: ${error.message}`
        );
    }

    try {

        const html =
            path.join(
                DEBUG_DIR,
                `${prefix}-${stamp}.html`
            );

        fs.writeFileSync(
            html,
            await page.content(),
            'utf8'
        );

        console.log(
            `📄 ${html}`
        );

    } catch (error) {

        console.log(
            `⚠️ HTML dump failed: ${error.message}`
        );
    }

    try {

        const text =
            path.join(
                DEBUG_DIR,
                `${prefix}-${stamp}.txt`
            );

        const body =
            await page.evaluate(
                () =>
                    document.body?.innerText ||
                    ''
            );

        fs.writeFileSync(
            text,
            body,
            'utf8'
        );

        console.log(
            `📝 ${text}`
        );

    } catch (error) {

        console.log(
            `⚠️ TXT dump failed: ${error.message}`
        );
    }
}

// ============================================================
// ONBOARDING
// ============================================================

async function isRealChatReady(
    page
) {

    try {

        return await page.evaluate(
            () => {

                function visible(
                    element
                ) {

                    if (!element) {
                        return false;
                    }

                    const rect =
                        element.getBoundingClientRect();

                    const style =
                        getComputedStyle(
                            element
                        );

                    return (
                        rect.width > 0 &&
                        rect.height > 0 &&
                        style.display !== 'none' &&
                        style.visibility !== 'hidden' &&
                        style.opacity !== '0'
                    );
                }

                const input =
                    document.querySelector(
                        'textarea[name="user-prompt"]'
                    ) ||
                    document.querySelector(
                        'textarea'
                    );

                if (
                    !visible(input)
                ) {
                    return false;
                }

                const onboarding =
                    [
                        ...document.querySelectorAll(
                            'button, [role="button"]'
                        )
                    ].some(
                        element => {

                            if (
                                !visible(element) ||
                                element.disabled
                            ) {
                                return false;
                            }

                            const text = (
                                element.innerText ||
                                element.textContent ||
                                element.getAttribute(
                                    'aria-label'
                                ) ||
                                ''
                            )
                                .replace(
                                    /\s+/g,
                                    ' '
                                )
                                .trim()
                                .toLowerCase();

                            return (
                                text === 'continue' ||
                                text === 'agree and continue' ||
                                text === 'accept and continue'
                            );
                        }
                    );

                return !onboarding;
            }
        );

    } catch (_) {

        return false;
    }
}

async function clickOnboarding(
    page
) {

    const clicked =
        await page.evaluate(
            () => {

                function norm(value) {
                    return String(
                        value || ''
                    )
                        .replace(
                            /\s+/g,
                            ' '
                        )
                        .trim()
                        .toLowerCase();
                }

                function visible(element) {

                    if (!element) {
                        return false;
                    }

                    const rect =
                        element.getBoundingClientRect();

                    const style =
                        getComputedStyle(
                            element
                        );

                    return (
                        rect.width > 0 &&
                        rect.height > 0 &&
                        style.display !== 'none' &&
                        style.visibility !== 'hidden' &&
                        style.opacity !== '0'
                    );
                }

                const priorities = [
                    'agree and continue',
                    'accept and continue',
                    'continue',
                    'agree'
                ];

                const buttons = [
                    ...document.querySelectorAll(
                        'button, [role="button"]'
                    )
                ];

                for (
                    const wanted of priorities
                ) {

                    for (
                        const button of buttons
                    ) {

                        if (
                            !visible(button) ||
                            button.disabled
                        ) {
                            continue;
                        }

                        const text =
                            norm(
                                button.innerText ||
                                button.textContent ||
                                button.getAttribute(
                                    'aria-label'
                                ) ||
                                ''
                            );

                        if (
                            text === wanted
                        ) {

                            button.click();

                            return wanted;
                        }
                    }
                }

                return null;
            }
        );

    if (clicked) {

        if (DEBUG_ENABLED) {
            console.log(
                `🖱️ Onboarding clicked: ${clicked}`
            );
        }

        await sleep(
            1200
        );

        return true;
    }

    return false;
}

async function handleOnboarding(
    page
) {

    if (DEBUG_ENABLED) {
        console.log(
            '🔍 Checking Duck.ai onboarding...'
        );
    }

    for (
        let i = 0;
        i < 10;
        i++
    ) {

        if (
            await isRealChatReady(
                page
            )
        ) {
            return true;
        }

        const clicked =
            await clickOnboarding(
                page
            );

        if (!clicked) {
            await sleep(500);
        }
    }

    return await isRealChatReady(
        page
    );
}

// ============================================================
// NAVIGATION RETRY
// ============================================================

async function navigateWithRetry(
    page
) {

    let lastError =
        null;

    for (
        let attempt = 1;
        attempt <= 4;
        attempt++
    ) {

        try {

            console.log(
                `🌍 Opening ${DUCK_URL} (attempt ${attempt}/4)...`
            );

            await page.goto(
                DUCK_URL,
                {
                    waitUntil:
                        'domcontentloaded',
                    timeout:
                        PAGE_TIMEOUT
                }
            );

            // Wait for main textarea to appear
            try {
                await page.waitForSelector(
                    'textarea',
                    {
                        timeout: 15000,
                        visible: true
                    }
                );
            } catch (_) {
                // Fallback to sleep
                await sleep(2500);
            }

            const ready =
                await handleOnboarding(
                    page
                );

            if (ready) {
                return true;
            }

            console.log(
                `⚠️ Duck.ai UI not ready after navigation attempt ${attempt}`
            );

        } catch (error) {

            lastError =
                error;

            console.log(
                `⚠️ Navigation attempt ${attempt} failed: ${error.message}`
            );
        }

        await sleep(
            1500 * attempt
        );
    }

    if (lastError) {
        throw lastError;
    }

    return false;
}

// ============================================================
// REQUEST SESSION
// ============================================================

async function createRequestSession() {

    const b =
        await getBrowser();

    let context =
        null;

    let page =
        null;

    try {

        if (
            typeof b.createBrowserContext ===
            'function'
        ) {

            context =
                await b.createBrowserContext();

            page =
                await context.newPage();

        } else if (
            typeof b.createIncognitoBrowserContext ===
            'function'
        ) {

            context =
                await b.createIncognitoBrowserContext();

            page =
                await context.newPage();

        } else {

            page =
                await b.newPage();
        }

        await setupPage(
            page
        );

        const ready =
            await navigateWithRetry(
                page
            );

        if (!ready) {

            await saveDebug(
                page,
                'chat-not-ready'
            );

            throw new Error(
                'Duck.ai chat UI is not ready'
            );
        }

        console.log(
            '✅ Chat UI ready'
        );

        return {
            context,
            page
        };

    } catch (error) {

        try {

            if (page) {
                await page.close();
            }

        } catch (_) {}

        try {

            if (context) {
                await context.close();
            }

        } catch (_) {}

        throw error;
    }
}

async function closeRequestSession(
    session
) {

    if (!session) {
        return;
    }

    try {

        if (
            session.page &&
            !session.page.isClosed()
        ) {

            await session.page.close();
        }

    } catch (_) {}

    try {

        if (
            session.context &&
            typeof session.context.close ===
                'function'
        ) {

            await session.context.close();
        }

    } catch (_) {}
}

// ============================================================
// MODEL SELECTION
// ============================================================

async function getCurrentModel(
    page
) {

    try {

        return await page.evaluate(
            () => {

                const button =
                    document.querySelector(
                        '[data-testid="model-picker-button"]'
                    );

                if (!button) {
                    return '';
                }

                return (
                    button.innerText ||
                    button.textContent ||
                    button.getAttribute(
                        'aria-label'
                    ) ||
                    ''
                ).trim();
            }
        );

    } catch (_) {

        return '';
    }
}

async function openModelPicker(
    page
) {

    const selectors = [
        '[data-testid="model-picker-button"]',
        'button[aria-label*="model" i]',
        'button[data-testid*="model" i]'
    ];

    for (
        const selector of selectors
    ) {

        try {

            const element =
                await page.$(
                    selector
                );

            if (!element) {
                continue;
            }

            await element.click();

            await sleep(
                600
            );

            return true;

        } catch (_) {}
    }

    return false;
}

async function findAndClickModel(
    page,
    labels
) {

    const wanted =
        labels.map(
            normalize
        );

    return await page.evaluate(
        values => {

            function norm(value) {
                return String(
                    value || ''
                )
                    .replace(
                        /\s+/g,
                        ' '
                    )
                    .trim()
                    .toLowerCase();
            }

            function visible(element) {

                const rect =
                    element.getBoundingClientRect();

                const style =
                    getComputedStyle(
                        element
                    );

                return (
                    rect.width > 0 &&
                    rect.height > 0 &&
                    style.display !== 'none' &&
                    style.visibility !== 'hidden'
                );
            }

            const elements = [];

            for (
                const selector of [
                    'button',
                    '[role="option"]',
                    '[role="menuitem"]',
                    'li'
                ]
            ) {

                elements.push(
                    ...document.querySelectorAll(
                        selector
                    )
                );
            }

            for (
                const element of elements
            ) {

                if (
                    element.disabled ||
                    !visible(element)
                ) {
                    continue;
                }

                const text =
                    norm(
                        element.innerText ||
                        element.textContent ||
                        ''
                    );

                if (!text) {
                    continue;
                }

                if (
                    values.some(
                        value =>
                            text === value
                    )
                ) {

                    element.click();

                    return text;
                }
            }

            for (
                const element of elements
            ) {

                if (
                    element.disabled ||
                    !visible(element)
                ) {
                    continue;
                }

                const text =
                    norm(
                        element.innerText ||
                        element.textContent ||
                        ''
                    );

                if (!text) {
                    continue;
                }

                const matched =
                    values.find(
                        value =>
                            text.includes(value)
                    );

                if (matched) {

                    element.click();

                    return text;
                }
            }

            return '';
        },
        wanted
    );
}

async function verifyModel(
    page,
    actualModel
) {

    const definition =
        MODELS[
            actualModel
        ];

    if (!definition) {
        return false;
    }

    const current =
        await getCurrentModel(
            page
        );

    const currentNorm =
        normalize(
            current
        );

    const accepted = [
        actualModel,
        ...(definition.labels || []),
        ...(definition.aliases || [])
    ].map(
        normalize
    );

    return accepted.some(
        value =>
            currentNorm === value ||
            currentNorm.includes(value) ||
            value.includes(currentNorm)
    );
}

async function selectModel(
    page,
    actualModel
) {

    const definition =
        MODELS[
            actualModel
        ];

    if (!definition) {
        throw new Error(
            `Unsupported model: ${actualModel}`
        );
    }

    const current =
        await getCurrentModel(
            page
        );

    if (DEBUG_ENABLED) {
        console.log(
            `🤖 Current model: ${
                current || 'unknown'
            }`
        );
    }

    if (
        await verifyModel(
            page,
            actualModel
        )
    ) {

        if (DEBUG_ENABLED) {
            console.log(
                `✅ Model already selected: ${actualModel}`
            );
        }

        return true;
    }

    const opened =
        await openModelPicker(
            page
        );

    if (!opened) {
        throw new Error(
            `Could not open model picker for ${actualModel}`
        );
    }

    if (DEBUG_ENABLED) {
        console.log(
            '🔽 Model picker opened'
        );
    }

    const clicked =
        await findAndClickModel(
            page,
            [
                ...definition.labels,
                ...definition.aliases
            ]
        );

    if (!clicked) {
        try {
            await page.keyboard.press(
                'Escape'
            );
        } catch (_) {}

        throw new Error(
            `Model option not found: ${actualModel}`
        );
    }

    if (DEBUG_ENABLED) {
        console.log(
            `🖱️ Model option clicked: ${clicked}`
        );
    }

    await sleep(900);

    for (
        let i = 0;
        i < 15;
        i++
    ) {

        if (
            await verifyModel(
                page,
                actualModel
            )
        ) {

            if (DEBUG_ENABLED) {
                console.log(
                    `✅ VERIFIED Duck.ai model: ${await getCurrentModel(
                        page
                    )}`
                );
            }

            return true;
        }

        await sleep(300);
    }

    throw new Error(
        `Model verification failed. Requested '${actualModel}', Duck.ai currently shows '${await getCurrentModel(
            page
        )}'.`
    );
}

// ============================================================
// TEXTAREA
// ============================================================

async function getTextarea(
    page
) {

    const selectors = [
        'textarea[name="user-prompt"][aria-label="Ask anything privately"]',
        'textarea[name="user-prompt"]',
        'textarea[aria-label="Ask anything privately"]',
        'textarea'
    ];

    for (
        const selector of selectors
    ) {

        try {

            const element =
                await page.$(
                    selector
                );

            if (!element) {
                continue;
            }

            const valid =
                await page.evaluate(
                    node => {

                        const rect =
                            node.getBoundingClientRect();

                        const style =
                            getComputedStyle(
                                node
                            );

                        return (
                            !node.disabled &&
                            rect.width > 0 &&
                            rect.height > 0 &&
                            style.display !== 'none' &&
                            style.visibility !== 'hidden'
                        );
                    },
                    element
                );

            if (valid) {
                return element;
            }

        } catch (_) {}
    }

    throw new Error(
        'Duck.ai textarea not found'
    );
}

async function getTextareaValue(
    page
) {

    try {

        return await page.$eval(
            'textarea[name="user-prompt"]',
            element =>
                element.value || ''
        );

    } catch (_) {

        try {

            return await page.$eval(
                'textarea',
                element =>
                    element.value || ''
            );

        } catch (_) {

            return '';
        }
    }
}

async function clearTextarea(
    page
) {

    await page.evaluate(
        () => {

            const textarea =
                document.querySelector(
                    'textarea[name="user-prompt"]'
                ) ||
                document.querySelector(
                    'textarea'
                );

            if (!textarea) {
                return;
            }

            const setter =
                Object.getOwnPropertyDescriptor(
                    HTMLTextAreaElement.prototype,
                    'value'
                )?.set;

            if (setter) {
                setter.call(
                    textarea,
                    ''
                );
            } else {
                textarea.value = '';
            }

            textarea.dispatchEvent(
                new Event(
                    'input',
                    {
                        bubbles: true
                    }
                )
            );

            textarea.dispatchEvent(
                new Event(
                    'change',
                    {
                        bubbles: true
                    }
                )
            );
        }
    );

    await sleep(150);
}

async function setTextareaText(
    page,
    text
) {

    const textarea =
        await getTextarea(
            page
        );

    await textarea.click();

    await clearTextarea(
        page
    );

    // Use native setter first for speed
    await page.evaluate(
        value => {

            const textarea =
                document.querySelector(
                    'textarea[name="user-prompt"]'
                ) ||
                document.querySelector(
                    'textarea'
                );

            if (!textarea) {
                return;
            }

            const setter =
                Object.getOwnPropertyDescriptor(
                    HTMLTextAreaElement.prototype,
                    'value'
                )?.set;

            if (setter) {
                setter.call(
                    textarea,
                    value
                );
            } else {
                textarea.value =
                    value;
            }

            textarea.dispatchEvent(
                new InputEvent(
                    'input',
                    {
                        bubbles: true,
                        inputType: 'insertText',
                        data: value
                    }
                )
            );

            textarea.dispatchEvent(
                new Event(
                    'change',
                    {
                        bubbles: true
                    }
                )
            );
        },
        text
    );

    await sleep(500);

    let current =
        await getTextareaValue(
            page
        );

    if (
        current === text
    ) {
        return;
    }

    if (DEBUG_ENABLED) {
        console.log(
            '⚠️ Native setter failed, falling back to type()'
        );
    }

    // Fallback to typing
    await textarea.type(
        text,
        {
            delay: 10
        }
    );

    await sleep(500);

    current =
        await getTextareaValue(
            page
        );

    if (
        current !== text
    ) {
        throw new Error(
            `Input mismatch: expected ${text.length} chars, got ${current.length}`
        );
    }
}

// ============================================================
// SEND
// ============================================================

async function findAskButton(
    page
) {

    const selectors = [
        'button[aria-label="Ask"]',
        'button[type="submit"][aria-label="Ask"]',
        'form button[type="submit"]',
        'button[type="submit"]'
    ];

    for (
        const selector of selectors
    ) {

        try {

            const element =
                await page.$(
                    selector
                );

            if (!element) {
                continue;
            }

            const state =
                await page.evaluate(
                    node => {

                        const rect =
                            node.getBoundingClientRect();

                        const style =
                            getComputedStyle(
                                node
                            );

                        return {
                            visible:
                                rect.width > 0 &&
                                rect.height > 0 &&
                                style.display !== 'none' &&
                                style.visibility !== 'hidden',

                            disabled:
                                !!node.disabled
                        };
                    },
                    element
                );

            if (
                state.visible
            ) {

                return {
                    element,
                    state,
                    selector
                };
            }

        } catch (_) {}
    }

    return null;
}

async function sendMessage(
    page,
    prompt
) {

    await setTextareaText(
        page,
        prompt
    );

    let current =
        await getTextareaValue(
            page
        );

    if (
        current !== prompt
    ) {
        throw new Error(
            `Input mismatch: expected ${prompt.length} chars, got ${current.length}`
        );
    }

    let ask = null;

    for (
        let i = 0;
        i < 30;
        i++
    ) {

        ask =
            await findAskButton(
                page
            );

        if (
            ask &&
            !ask.state.disabled
        ) {
            break;
        }

        await sleep(200);
    }

    if (
        ask &&
        !ask.state.disabled
    ) {

        if (DEBUG_ENABLED) {
            console.log(
                `📤 Clicking Ask: ${ask.selector}`
            );
        }

        await ask.element.click();

        await sleep(
            1000
        );

        current =
            await getTextareaValue(
                page
            );

        if (
            current !== prompt
        ) {
            if (DEBUG_ENABLED) {
                console.log(
                    '✅ Message submitted with Ask'
                );
            }

            return;
        }
    }

    if (DEBUG_ENABLED) {
        console.log(
            '⚠️ Ask did not submit, trying Enter...'
        );
    }

    const textarea =
        await getTextarea(
            page
        );

    await textarea.focus();

    await textarea.press(
        'Enter'
    );

    await sleep(
        1000
    );

    current =
        await getTextareaValue(
            page
        );

    if (
        current !== prompt
    ) {

        if (DEBUG_ENABLED) {
            console.log(
                '✅ Message submitted with Enter'
            );
        }

        return;
    }

    throw new Error(
        'Message submission failed'
    );
}

// ============================================================
// RESPONSE
// ============================================================

async function getBodyText(
    page
) {

    try {

        return await page.evaluate(
            () =>
                document.body?.innerText ||
                ''
        );

    } catch (_) {

        return '';
    }
}

async function getAssistantCandidates(
    page
) {

    return await page.evaluate(
        () => {

            const selectors = [
                '[data-message-role="assistant"]',
                '[data-role="assistant"]',
                '[data-testid*="assistant"]',
                '[data-testid*="assistant-message"]'
            ];

            const result = [];

            for (
                const selector of selectors
            ) {

                let nodes = [];

                try {

                    nodes = [
                        ...document.querySelectorAll(
                            selector
                        )
                    ];

                } catch (_) {
                    continue;
                }

                for (
                    const node of nodes
                ) {

                    const text =
                        (
                            node.innerText ||
                            node.textContent ||
                            ''
                        ).trim();

                    if (!text) {
                        continue;
                    }

                    const signature = [
                        node.getAttribute(
                            'data-testid'
                        ) || '',

                        node.getAttribute(
                            'data-role'
                        ) || '',

                        node.getAttribute(
                            'data-message-role'
                        ) || '',

                        node.getAttribute(
                            'aria-label'
                        ) || ''
                    ]
                        .join(' ')
                        .toLowerCase();

                    if (
                        signature.includes('user') ||
                        signature.includes('human')
                    ) {
                        continue;
                    }

                    result.push({
                        selector,
                        text
                    });
                }
            }

            const seen =
                new Set();

            return result.filter(
                item => {

                    const key =
                        item.text
                            .replace(
                                /\s+/g,
                                ' '
                            )
                            .trim();

                    if (
                        seen.has(key)
                    ) {
                        return false;
                    }

                    seen.add(key);

                    return true;
                }
            );
        }
    );
}

function cleanDuckResponse(
    raw,
    actualModel
) {

    let text =
        cleanText(
            raw
        );

    if (!text) {
        return '';
    }

    const footerMarkers = [
        'Duck.ai works best in our private and free DuckDuckGo app!',
        'Duck.ai works best in our private and free DuckDuckGo app',
        'All chats are private. AI can make mistakes.',
        'All chats are private.',
        'Download',
        'Learn More',
        'Learn more',
        'How Duck.ai Works',
        'Chat Suggestions',
        'Create & Edit Images'
    ];

    for (
        const marker of footerMarkers
    ) {

        const index =
            text.indexOf(
                marker
            );

        if (
            index !== -1
        ) {

            text =
                text
                    .slice(
                        0,
                        index
                    )
                    .trim();
        }
    }

    const definition =
        MODELS[
            actualModel
        ];

    const modelNames =
        definition
            ? [
                actualModel,
                ...(definition.labels || []),
                ...(definition.aliases || [])
            ].map(normalize)
            : [];

    const transient = new Set([
        'generating response',
        'stop generating',
        'tools',
        'fast',
        'ask',
        'download',
        'learn more',
        'all chats are private. ai can make mistakes.'
    ]);

    let lines =
        text
            .split('\n')
            .map(
                line =>
                    line.trim()
            )
            .filter(Boolean);

    while (
        lines.length > 0
    ) {

        const first =
            normalize(
                lines[0]
            );

        if (
            modelNames.includes(first) ||
            transient.has(first)
        ) {

            lines.shift();

        } else {

            break;
        }
    }

    lines =
        lines.filter(
            line =>
                !transient.has(
                    normalize(line)
                )
        );

    text =
        cleanText(
            lines.join('\n')
        );

    for (
        const marker of footerMarkers
    ) {

        const index =
            text.indexOf(
                marker
            );

        if (
            index !== -1
        ) {

            text =
                text
                    .slice(
                        0,
                        index
                    )
                    .trim();
        }
    }

    return cleanText(
        text
    );
}

async function isGenerating(
    page
) {

    try {

        return await page.evaluate(
            () => {

                const selectors = [
                    'button[aria-label="Stop generating"]',
                    'button[aria-label*="Stop generating" i]',
                    '[data-testid*="stop" i]'
                ];

                return selectors.some(
                    selector => {

                        const element =
                            document.querySelector(
                                selector
                            );

                        if (!element) {
                            return false;
                        }

                        const rect =
                            element.getBoundingClientRect();

                        const style =
                            getComputedStyle(
                                element
                            );

                        return (
                            rect.width > 0 &&
                            rect.height > 0 &&
                            style.display !== 'none' &&
                            style.visibility !== 'hidden'
                        );
                    }
                );
            }
        );

    } catch (_) {

        return false;
    }
}

function extractAfterPrompt(
    body,
    prompt
) {

    const text =
        cleanText(
            body
        );

    if (
        !text ||
        !prompt
    ) {
        return '';
    }

    const index =
        text.lastIndexOf(
            prompt
        );

    if (
        index === -1
    ) {
        return '';
    }

    return text
        .slice(
            index + prompt.length
        )
        .trim();
}

async function waitForResponse(
    page,
    prompt,
    lastUser,
    actualModel,
    oldCandidates,
    isClientClosed
) {

    const started =
        Date.now();

    const oldSet =
        new Set(
            oldCandidates.map(
                item =>
                    normalize(
                        item.text
                    )
            )
        );

    let lastResponse = '';
    let lastNormalized = '';
    let stable = 0;
    let seen = false;

    while (
        Date.now() - started <
        RESPONSE_TIMEOUT
    ) {

        if (
            isClientClosed &&
            isClientClosed()
        ) {
            throw new Error(
                'Client disconnected'
            );
        }

        await sleep(
            POLL_INTERVAL
        );

        let candidates = [];

        try {

            candidates =
                await getAssistantCandidates(
                    page
                );

        } catch (_) {}

        let response =
            '';

        const values =
            candidates
                .map(
                    item =>
                        cleanText(
                            item.text
                        )
                )
                .filter(Boolean)
                .sort(
                    (a, b) =>
                        b.length -
                        a.length
                );

        for (
            const candidate of values
        ) {

            const normalized =
                normalize(
                    candidate
                );

            if (
                oldSet.has(
                    normalized
                )
            ) {
                continue;
            }

            if (
                normalized === normalize(prompt) ||
                normalized === normalize(lastUser)
            ) {
                continue;
            }

            response =
                candidate;

            break;
        }

        if (!response) {

            const body =
                await getBodyText(
                    page
                );

            response =
                extractAfterPrompt(
                    body,
                    prompt
                );
        }

        response =
            cleanDuckResponse(
                response,
                actualModel
            );

        if (!response) {
            continue;
        }

        seen = true;

        const normalized =
            normalize(
                response
            );

        if (
            normalized ===
            lastNormalized
        ) {

            stable++;

        } else {

            stable = 0;

            lastResponse =
                response;

            lastNormalized =
                normalized;

            if (DEBUG_ENABLED) {
                console.log(
                    `🧩 Response update [${response.length} chars]: ${response.slice(
                        0,
                        220
                    )}`
                );
            }
        }

        const generating =
            await isGenerating(
                page
            );

        if (
            seen &&
            !generating &&
            stable >= 2
        ) {

            return cleanDuckResponse(
                lastResponse,
                actualModel
            );
        }

        if (
            stable >= STABLE_POLLS
        ) {

            return cleanDuckResponse(
                lastResponse,
                actualModel
            );
        }
    }

    if (lastResponse) {
        return cleanDuckResponse(
            lastResponse,
            actualModel
        );
    }

    throw new Error(
        `Empty response after ${RESPONSE_TIMEOUT}ms`
    );
}

// ============================================================
// EXECUTE REQUEST
// ============================================================

async function executeDuckRequest(
    messages,
    actualModel,
    requestId,
    isClientClosed
) {

    let session = null;

    try {

        session =
            await createRequestSession();

        const page =
            session.page;

        await selectModel(
            page,
            actualModel
        );

        if (
            isClientClosed &&
            isClientClosed()
        ) {
            throw new Error(
                'Client disconnected'
            );
        }

        if (
            !(await verifyModel(
                page,
                actualModel
            ))
        ) {

            throw new Error(
                `Model verification failed before sending: ${actualModel}`
            );
        }

        const prompt =
            buildDuckPrompt(
                messages
            );

        if (!prompt.trim()) {
            throw new Error(
                'Generated Duck.ai prompt is empty'
            );
        }

        console.log(
            `🤖 [${requestId}] Verified model: ${actualModel}`
        );

        console.log(
            `📝 [${requestId}] Prompt length: ${prompt.length}`
        );

        const oldCandidates =
            await getAssistantCandidates(
                page
            );

        await sendMessage(
            page,
            prompt
        );

        const result =
            await waitForResponse(
                page,
                prompt,
                getLastUserMessage(
                    messages
                ),
                actualModel,
                oldCandidates,
                isClientClosed
            );

        const cleaned =
            cleanDuckResponse(
                result,
                actualModel
            );

        if (!cleaned) {
            throw new Error(
                'Empty response after cleanup'
            );
        }

        return cleaned;

    } finally {

        await closeRequestSession(
            session
        );
    }
}

// ============================================================
// RETRY
// ============================================================

async function chatWithDuckAI(
    messages,
    actualModel,
    requestId,
    isClientClosed
) {

    let lastError = null;

    for (
        let attempt = 1;
        attempt <= MAX_RETRIES + 1;
        attempt++
    ) {

        if (
            isClientClosed &&
            isClientClosed()
        ) {
            throw new Error(
                'Client disconnected'
            );
        }

        try {

            console.log('');
            console.log(
                `================ REQUEST ${requestId} ATTEMPT ${attempt} ================`
            );

            return await executeDuckRequest(
                messages,
                actualModel,
                requestId,
                isClientClosed
            );

        } catch (error) {

            lastError =
                error;

            console.error(
                `❌ [${requestId}] Attempt ${attempt} failed: ${error.message}`
            );

            if (
                attempt <= MAX_RETRIES
            ) {

                const delay =
                    attempt * 1500;

                console.log(
                    `🔄 [${requestId}] Retrying in ${delay}ms...`
                );

                await sleep(
                    delay
                );
            }
        }
    }

    throw (
        lastError ||
        new Error(
            'Duck.ai request failed'
        )
    );
}

// ============================================================
// USAGE
// ============================================================

function estimateUsage(
    messages,
    response
) {

    const promptChars =
        messages.reduce(
            (
                total,
                message
            ) =>
                total +
                contentToText(
                    message.content
                ).length,
            0
        );

    const completionChars =
        String(
            response || ''
        ).length;

    const promptTokens =
        Math.max(
            1,
            Math.ceil(
                promptChars / 4
            )
        );

    const completionTokens =
        Math.max(
            1,
            Math.ceil(
                completionChars / 4
            )
        );

    return {
        prompt_tokens:
            promptTokens,

        completion_tokens:
            completionTokens,

        total_tokens:
            promptTokens +
            completionTokens
    };
}

// ============================================================
// OPENAI RESPONSE (Chat Completions)
// ============================================================

function createChatCompletion(
    model,
    response,
    messages,
    requestId,
    responseTime
) {

    return {

        id:
            makeId(
                'chatcmpl'
            ),

        object:
            'chat.completion',

        created:
            Math.floor(
                Date.now() / 1000
            ),

        model,

        choices: [
            {
                index: 0,

                message: {
                    role:
                        'assistant',

                    content:
                        response
                },

                finish_reason:
                    'stop'
            }
        ],

        usage:
            estimateUsage(
                messages,
                response
            ),

        _proxy: {

            request_id:
                requestId,

            actual_model:
                model,

            response_time_ms:
                responseTime
        }
    };
}

// ============================================================
// OPENAI RESPONSES API
// ============================================================

function createResponsesCompletion(
    model,
    response,
    messages,
    requestId,
    responseTime
) {
    const usage = estimateUsage(messages, response);
    
    return {
        id: makeId('resp'),
        object: 'response',
        created_at: Math.floor(Date.now() / 1000),
        status: 'completed',
        model,
        output: [
            {
                type: 'message',
                id: makeId('msg'),
                status: 'completed',
                role: 'assistant',
                content: [
                    {
                        type: 'output_text',
                        text: response,
                        annotations: []
                    }
                ]
            }
        ],
        usage: {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            input_tokens_details: {
                cached_tokens: 0
            },
            output_tokens_details: {
                reasoning_tokens: 0
            }
        },
        metadata: {
            request_id: requestId,
            actual_model: model,
            response_time_ms: responseTime
        }
    };
}

async function streamResponsesCompletion(
    res,
    model,
    response,
    messages
) {
    const chunks = response.match(/[\s\S]{1,90}/g) || [response];
    
    if (!res.headersSent) {
        res.status(200);
        res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
        res.setHeader('Cache-Control', 'no-cache, no-transform');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');
        res.flushHeaders?.();
    }
    
    const seq = () => Math.floor(Date.now() / 1000);
    
    // Initial response object
    const initialResponse = createResponsesCompletion(
        model,
        response,
        messages,
        'stream',
        0
    );
    
    // response.created
    res.write(`event: response.created\ndata: ${JSON.stringify({
        type: 'response.created',
        sequence_number: seq(),
        response: {
            ...initialResponse,
            status: 'in_progress',
            output: []
        }
    })}\n\n`);
    
    // response.in_progress
    res.write(`event: response.in_progress\ndata: ${JSON.stringify({
        type: 'response.in_progress',
        sequence_number: seq(),
        response: {
            ...initialResponse,
            status: 'in_progress',
            output: []
        }
    })}\n\n`);
    
    // output_item.added
    const messageId = makeId('msg');
    res.write(`event: response.output_item.added\ndata: ${JSON.stringify({
        type: 'response.output_item.added',
        sequence_number: seq(),
        output_index: 0,
        item: {
            id: messageId,
            type: 'message',
            status: 'in_progress',
            role: 'assistant',
            content: []
        }
    })}\n\n`);
    
    // content_part.added
    res.write(`event: response.content_part.added\ndata: ${JSON.stringify({
        type: 'response.content_part.added',
        sequence_number: seq(),
        item_id: messageId,
        output_index: 0,
        content_index: 0,
        part: {
            type: 'output_text',
            text: '',
            annotations: []
        }
    })}\n\n`);
    
    // Send text deltas
    for (const chunk of chunks) {
        if (res.destroyed) return;
        
        res.write(`event: response.output_text.delta\ndata: ${JSON.stringify({
            type: 'response.output_text.delta',
            sequence_number: seq(),
            item_id: messageId,
            output_index: 0,
            content_index: 0,
            delta: chunk
        })}\n\n`);
        
        if (STREAM_DELAY > 0) {
            await sleep(STREAM_DELAY);
        }
    }
    
    // content_part.done
    res.write(`event: response.content_part.done\ndata: ${JSON.stringify({
        type: 'response.content_part.done',
        sequence_number: seq(),
        item_id: messageId,
        output_index: 0,
        content_index: 0,
        part: {
            type: 'output_text',
            text: response,
            annotations: []
        }
    })}\n\n`);
    
    // output_item.done
    res.write(`event: response.output_item.done\ndata: ${JSON.stringify({
        type: 'response.output_item.done',
        sequence_number: seq(),
        output_index: 0,
        item: {
            id: messageId,
            type: 'message',
            status: 'completed',
            role: 'assistant',
            content: [
                {
                    type: 'output_text',
                    text: response,
                    annotations: []
                }
            ]
        }
    })}\n\n`);
    
    // response.completed
    const finalResponse = createResponsesCompletion(
        model,
        response,
        messages,
        'stream',
        0
    );
    
    res.write(`event: response.completed\ndata: ${JSON.stringify({
        type: 'response.completed',
        sequence_number: seq(),
        response: finalResponse
    })}\n\n`);
    
    res.write('data: [DONE]\n\n');
    res.end();
}

// ============================================================
// SSE (Chat Completions)
// ============================================================

function sendSSE(
    res,
    data
) {

    res.write(
        `data: ${JSON.stringify(
            data
        )}\n\n`
    );
}

async function streamCompletion(res, model, response, messages) {
    const id = makeId('chatcmpl');
    const created = Math.floor(Date.now() / 1000);
    const usage = estimateUsage(messages, response);

    if (!res.headersSent) {
        res.status(200);
        res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
        res.setHeader('Cache-Control', 'no-cache, no-transform');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');
        res.flushHeaders?.();
    }

    // 1) First chunk with role
    sendSSE(res, {
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [{
            index: 0,
            delta: { role: 'assistant' },
            finish_reason: null
        }]
    });

    // 2) Single content chunk (whole response)
    sendSSE(res, {
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [{
            index: 0,
            delta: { content: response },
            finish_reason: null
        }]
    });

    // 3) Final chunk with finish_reason and usage
    sendSSE(res, {
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [{
            index: 0,
            delta: {},
            finish_reason: 'stop'
        }],
        usage
    });

    res.write('data: [DONE]\n\n');
    res.end();
}

// ============================================================
// AUTH
// ============================================================

function checkAuth(
    req
) {

    if (!REQUIRE_API_KEY) {
        return true;
    }

    if (!API_KEY) {
        return false;
    }

    const authorization =
        req.get(
            'authorization'
        ) || '';

    const xApiKey =
        req.get(
            'x-api-key'
        ) || '';

    if (
        authorization
            .toLowerCase()
            .startsWith(
                'bearer '
            )
    ) {

        return (
            authorization
                .slice(7)
                .trim() ===
            API_KEY
        );
    }

    return (
        xApiKey ===
        API_KEY
    );
}

// ============================================================
// ERROR
// ============================================================

function sendOpenAIError(
    res,
    status,
    message,
    type,
    code = null,
    param = null
) {

    return res
        .status(status)
        .json({
            error: {
                message,
                type,
                param,
                code
            }
        });
}

// ============================================================
// CORS
// ============================================================

app.use(
    (
        req,
        res,
        next
    ) => {

        res.setHeader(
            'Access-Control-Allow-Origin',
            '*'
        );

        res.setHeader(
            'Access-Control-Allow-Methods',
            'GET, POST, OPTIONS'
        );

        res.setHeader(
            'Access-Control-Allow-Headers',
            'Content-Type, Authorization, X-API-Key'
        );

        if (
            req.method ===
            'OPTIONS'
        ) {

            return res
                .status(204)
                .end();
        }

        next();
    }
);

// ============================================================
// ROOT
// ============================================================

app.get(
    '/',
    (
        req,
        res
    ) => {

        res.json({
            object: 'api',

            name:
                'Duck.ai OpenAI-Compatible Proxy',

            version:
                'railway-1.2.0',

            host:
                HOST,

            port:
                PORT,

            concurrency:
                MAX_CONCURRENCY,

            endpoints: {
                models:
                    '/v1/models',

                chat:
                    '/v1/chat/completions',

                responses:
                    '/v1/responses',

                health:
                    '/health'
            }
        });
    }
);

// ============================================================
// HEALTH
// ============================================================

app.get(
    '/health',
    (
        req,
        res
    ) => {

        res.json({

            status:
                'ok',

            browser:
                browser &&
                browser.isConnected()
                    ? 'connected'
                    : 'disconnected',

            concurrency: {

                max:
                    pool.size,

                active:
                    pool.active,

                available:
                    pool.available,

                waiting:
                    pool.waiting.length,

                completed:
                    pool.completedCount,

                failed:
                    pool.failedCount
            },

            environment: {
                node:
                    process.version,

                platform:
                    process.platform,

                arch:
                    process.arch
            },

            timestamp:
                new Date()
                    .toISOString()
        });
    }
);

// ============================================================
// MODELS
// ============================================================

app.get(
    [
        '/v1/models',
        '/models'
    ],
    (
        req,
        res
    ) => {

        const created =
            Math.floor(
                Date.now() / 1000
            );

        res.json({

            object:
                'list',

            data:
                Object.values(
                    MODELS
                ).map(
                    model => ({

                        id:
                            model.id,

                        object:
                            'model',

                        created,

                        owned_by:
                            'duck.ai',

                        permission: [],

                        root:
                            model.id,

                        parent:
                            null
                    })
                )
        });
    }
);

// ============================================================
// HANDLE RESPONSES API
// ============================================================

async function handleResponsesCompletions(req, res) {
    // Convert Responses API format to Chat Completions format
    if (req.body && !req.body.messages && req.body.input) {
        const input = req.body.input;
        
        if (typeof input === 'string') {
            req.body.messages = [
                { role: 'user', content: input }
            ];
        } else if (Array.isArray(input)) {
            req.body.messages = input.map(item => {
                if (typeof item === 'string') {
                    return { role: 'user', content: item };
                }
                if (item && item.role && item.content) {
                    return item;
                }
                if (item && item.type === 'message' && item.content) {
                    const textContent = Array.isArray(item.content)
                        ? item.content.map(c => c.text || '').join('\n')
                        : item.content;
                    return {
                        role: item.role || 'user',
                        content: textContent
                    };
                }
                return { role: 'user', content: JSON.stringify(item) };
            });
        }
    }
    
    // Add instructions as system message if not present
    if (req.body && req.body.instructions && !req.body.messages?.some(m => m.role === 'system')) {
        req.body.messages = [
            { role: 'system', content: req.body.instructions },
            ...(req.body.messages || [])
        ];
    }
    
    return handleChatCompletions(req, res);
}

// ============================================================
// CHAT COMPLETIONS HANDLER
// ============================================================

async function handleChatCompletions(
    req,
    res
) {

    const started =
        Date.now();

    const requestId =
        makeId(
            'req'
        );

    let clientClosed = false;

    res.on('close', () => {
        clientClosed = true;
    });

    const isClientClosed = () => clientClosed;

    const isResponsesAPI = req.path === '/v1/responses';

    /*
     * AUTH
     */

    if (
        !checkAuth(req)
    ) {

        return sendOpenAIError(
            res,

            401,

            'Incorrect API key provided.',

            'authentication_error',

            'invalid_api_key'
        );
    }

    /*
     * Acquire concurrency slot.
     */

    await pool.acquire();

    try {

        const body =
            req.body &&
            typeof req.body ===
                'object'
                ? req.body
                : {};

        let messages =
            body.messages;

        /*
         * Legacy:
         * /api/chat
         */

        if (
            !Array.isArray(
                messages
            )
        ) {

            if (
                body.message
            ) {

                messages = [
                    {
                        role:
                            'user',

                        content:
                            String(
                                body.message
                            )
                    }
                ];

            } else if (
                body.input
            ) {

                messages = [
                    {
                        role:
                            'user',

                        content:
                            typeof body.input ===
                                'string'
                                ? body.input
                                : JSON.stringify(
                                    body.input
                                )
                    }
                ];

            } else {

                return sendOpenAIError(
                    res,

                    400,

                    'messages must be a non-empty array.',

                    'invalid_request_error',

                    'invalid_messages',

                    'messages'
                );
            }
        }

        try {

            validateMessages(
                messages
            );

        } catch (
            error
        ) {

            return sendOpenAIError(
                res,

                error.status ||
                    400,

                error.message,

                'invalid_request_error',

                error.code ||
                    'invalid_request',

                'messages'
            );
        }

        const requestedModel =
            String(
                body.model ||
                DEFAULT_MODEL
            ).trim();

        let modelInfo;

        try {

            modelInfo =
                resolveModel(
                    requestedModel
                );

        } catch (
            error
        ) {

            return sendOpenAIError(
                res,

                400,

                error.message,

                'invalid_request_error',

                'model_not_found',

                'model'
            );
        }

        const actualModel =
            modelInfo.actual;

        const stream =
            body.stream === true;

        console.log('');
        console.log(
            '================================================'
        );

        console.log(
            `📨 Request: ${requestId}`
        );

        console.log(
            `API Type: ${isResponsesAPI ? 'Responses' : 'Chat Completions'}`
        );

        console.log(
            `Requested model: ${requestedModel}`
        );

        console.log(
            `Actual model: ${actualModel}`
        );

        console.log(
            `Messages: ${messages.length}`
        );

        console.log(
            `Stream: ${stream}`
        );

        console.log(
            `Pool: ${pool.active}/${pool.size} active, ${pool.waiting.length} waiting`
        );

        console.log(
            `Latest user: ${getLastUserMessage(
                messages
            ).slice(0, 200)}`
        );

        console.log(
            '================================================'
        );

        /*
         * STREAM - Send headers immediately
         */

        if (stream) {

            res.status(200);

            res.setHeader(
                'Content-Type',
                'text/event-stream; charset=utf-8'
            );

            res.setHeader(
                'Cache-Control',
                'no-cache, no-transform'
            );

            res.setHeader(
                'Connection',
                'keep-alive'
            );

            res.setHeader(
                'X-Accel-Buffering',
                'no'
            );

            res.flushHeaders?.();

            // Optional keep-alive comments
            let keepAliveTimer = null;
            if (KEEP_ALIVE_INTERVAL > 0) {
                keepAliveTimer = setInterval(() => {
                    if (!res.destroyed) {
                        res.write(
                            ': keep-alive\n\n'
                        );
                    }
                }, KEEP_ALIVE_INTERVAL);
            }

            try {

                const response =
                    await chatWithDuckAI(
                        messages,
                        actualModel,
                        requestId,
                        isClientClosed
                    );

                if (keepAliveTimer) {
                    clearInterval(
                        keepAliveTimer
                    );
                }

                pool.completed();

                const responseTime =
                    Date.now() -
                    started;

                console.log(
                    `⚡ [${requestId}] Completed in ${responseTime}ms, Response length: ${response.length}`
                );

                if (isResponsesAPI) {
                    await streamResponsesCompletion(
                        res,
                        actualModel,
                        response,
                        messages
                    );
                } else {
                    await streamCompletion(
                        res,
                        actualModel,
                        response,
                        messages
                    );
                }

            } catch (error) {

                if (keepAliveTimer) {
                    clearInterval(
                        keepAliveTimer
                    );
                }

                pool.failed();

                console.error(
                    `❌ [${requestId}] ${
                        error.stack ||
                        error.message
                    }`
                );

                if (!res.destroyed) {

                    try {

                        if (isResponsesAPI) {
                            // Send error in Responses API format
                            res.write(`event: response.failed\ndata: ${JSON.stringify({
                                type: 'response.failed',
                                sequence_number: Math.floor(Date.now() / 1000),
                                response: {
                                    id: makeId('resp'),
                                    object: 'response',
                                    created_at: Math.floor(Date.now() / 1000),
                                    status: 'failed',
                                    model: actualModel,
                                    output: [],
                                    error: {
                                        code: error.code || 'proxy_error',
                                        message: error.message || 'Internal server error'
                                    }
                                }
                            })}\n\n`);
                            
                            res.write('data: [DONE]\n\n');
                            res.end();
                        } else {
                            sendSSE(
                                res,
                                {
                                    error: {
                                        message:
                                            error.message ||
                                            'Internal server error',

                                        type:
                                            'server_error',

                                        code:
                                            error.code ||
                                            'proxy_error'
                                    }
                                }
                            );

                            res.write(
                                'data: [DONE]\n\n'
                            );

                            res.end();
                        }

                    } catch (_) {}
                }
            }

            return;
        }

        /*
         * NON-STREAM
         */

        const response =
            await chatWithDuckAI(
                messages,
                actualModel,
                requestId,
                isClientClosed
            );

        pool.completed();

        const responseTime =
            Date.now() -
            started;

        console.log(
            `⚡ [${requestId}] Completed in ${responseTime}ms, Response length: ${response.length}`
        );

        if (isResponsesAPI) {
            return res.json(
                createResponsesCompletion(
                    actualModel,
                    response,
                    messages,
                    requestId,
                    responseTime
                )
            );
        }

        return res.json(
            createChatCompletion(
                actualModel,
                response,
                messages,
                requestId,
                responseTime
            )
        );

    } catch (
        error
    ) {

        pool.failed();

        const status =
            Number(
                error.status ||
                500
            );

        console.error(
            `❌ [${requestId}] ${
                error.stack ||
                error.message
            }`
        );

        if (!res.headersSent) {

            return sendOpenAIError(
                res,

                status,

                error.message ||
                    'Internal server error',

                status >= 500
                    ? 'server_error'
                    : 'invalid_request_error',

                error.code ||
                    'proxy_error'
            );
        }

    } finally {

        pool.release();

        if (DEBUG_ENABLED) {
            console.log(
                `📊 Pool: active=${pool.active}/${pool.size}, waiting=${pool.waiting.length}`
            );
        }
    }
}

// ============================================================
// ROUTES
// ============================================================

app.post(
    [
        '/v1/chat/completions',
        '/chat/completions',
        '/api/chat'
    ],
    handleChatCompletions
);

app.post(
    '/v1/responses',
    handleResponsesCompletions
);

// ============================================================
// EXPRESS ERROR
// ============================================================

app.use(
    (
        error,
        req,
        res,
        next
    ) => {

        console.error(
            '❌ Express error:',
            error
        );

        if (
            res.headersSent
        ) {

            return next(
                error
            );
        }

        return sendOpenAIError(
            res,

            500,

            error.message ||
                'Internal server error',

            'server_error',

            'internal_error'
        );
    }
);

// ============================================================
// SHUTDOWN
// ============================================================

async function shutdown(
    signal
) {

    if (
        shuttingDown
    ) {
        return;
    }

    shuttingDown =
        true;

    console.log(
        `\n🛑 ${signal} received`
    );

    console.log(
        `⏳ Active requests: ${pool.active}`
    );

    const deadline =
        Date.now() +
        15000;

    while (
        pool.active > 0 &&
        Date.now() < deadline
    ) {

        await sleep(
            250
        );
    }

    try {

        if (browser) {
            await browser.close();
        }

    } catch (_) {}

    browser =
        null;

    console.log(
        '✅ Shutdown complete'
    );

    process.exit(
        0
    );
}

process.on(
    'SIGINT',
    () => shutdown('SIGINT')
);

process.on(
    'SIGTERM',
    () => shutdown('SIGTERM')
);

process.on(
    'uncaughtException',
    error => {

        console.error(
            '💥 Uncaught exception:',
            error
        );

        process.exit(1);
    }
);

process.on(
    'unhandledRejection',
    error => {

        console.error(
            '💥 Unhandled rejection:',
            error
        );
    }
);

// ============================================================
// START
// ============================================================

async function startServer() {

    console.log('');
    console.log(
        '=============================================='
    );

    console.log(
        '🚀 Duck.ai OpenAI-Compatible Proxy'
    );

    console.log(
        '=============================================='
    );

    console.log(
        `🌍 Platform: ${process.platform}`
    );

    console.log(
        `🌍 Architecture: ${process.arch}`
    );

    console.log(
        `⚡ Max concurrency: ${MAX_CONCURRENCY}`
    );

    console.log(
        `🔌 Host: ${HOST}`
    );

    console.log(
        `🔌 Port: ${PORT}`
    );

    console.log(
        `⏱️ Response timeout: ${RESPONSE_TIMEOUT}ms`
    );

    console.log(
        `🔁 Retries: ${MAX_RETRIES}`
    );

    console.log(
        `🔐 API key required: ${REQUIRE_API_KEY}`
    );

    console.log(
        `💬 Keep-alive interval: ${KEEP_ALIVE_INTERVAL}ms (0=disabled)`
    );

    try {

        await getBrowser();

        app.listen(
            PORT,
            HOST,
            () => {

                console.log('');
                console.log(
                    '=============================================='
                );

                console.log(
                    `🌟 Server listening on http://${HOST}:${PORT}`
                );

                console.log(
                    `❤️ Health: /health`
                );

                console.log(
                    `📚 Models: /v1/models`
                );

                console.log(
                    `🤖 Chat: /v1/chat/completions`
                );

                console.log(
                    `📡 Responses: /v1/responses`
                );

                console.log(
                    '=============================================='
                );

                console.log('');
            }
        );

    } catch (
        error
    ) {

        console.error(
            '❌ Failed to start:',
            error
        );

        process.exit(
            1
        );
    }
}

startServer();
