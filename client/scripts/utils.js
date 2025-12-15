/**
 * Taiwan CCTV Viewer - Utility Functions
 * Shared JavaScript for all pages
 */

// ========================================
// Toast Notification System
// ========================================
const Toast = {
    container: null,

    init() {
        if (this.container) return;

        this.container = document.createElement('div');
        this.container.className = 'toast-container';
        this.container.setAttribute('role', 'alert');
        this.container.setAttribute('aria-live', 'polite');
        document.body.appendChild(this.container);
    },

    show(message, type = 'info', options = {}) {
        this.init();

        const { title, duration = 4000, closeable = true } = options;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;

        const icons = {
            success: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="10" fill="#10b981"/><path d="M6 10l3 3 5-6" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
            error: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="10" fill="#ef4444"/><path d="M7 7l6 6M13 7l-6 6" stroke="white" stroke-width="2" stroke-linecap="round"/></svg>',
            warning: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="10" fill="#f59e0b"/><path d="M10 6v5M10 14v.01" stroke="white" stroke-width="2" stroke-linecap="round"/></svg>',
            info: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="10" fill="#0066cc"/><path d="M10 9v5M10 6v.01" stroke="white" stroke-width="2" stroke-linecap="round"/></svg>'
        };

        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || icons.info}</span>
            <div class="toast-content">
                ${title ? `<div class="toast-title">${title}</div>` : ''}
                <div class="toast-message">${message}</div>
            </div>
            ${closeable ? '<button class="toast-close" aria-label="Close notification">&times;</button>' : ''}
        `;

        if (closeable) {
            toast.querySelector('.toast-close').addEventListener('click', () => {
                this.dismiss(toast);
            });
        }

        this.container.appendChild(toast);

        // Auto dismiss
        if (duration > 0) {
            setTimeout(() => {
                this.dismiss(toast);
            }, duration);
        }

        return toast;
    },

    dismiss(toast) {
        if (!toast || !toast.parentNode) return;

        toast.classList.add('exiting');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    },

    success(message, options = {}) {
        return this.show(message, 'success', options);
    },

    error(message, options = {}) {
        return this.show(message, 'error', options);
    },

    warning(message, options = {}) {
        return this.show(message, 'warning', options);
    },

    info(message, options = {}) {
        return this.show(message, 'info', options);
    }
};

// ========================================
// Debounce Function
// ========================================
function debounce(func, wait, immediate = false) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func.apply(this, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(this, args);
    };
}

// ========================================
// Throttle Function
// ========================================
function throttle(func, limit) {
    let inThrottle;
    return function executedFunction(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ========================================
// Connection Status Manager
// ========================================
const ConnectionStatus = {
    element: null,
    status: 'disconnected',
    lastUpdate: null,

    init(containerId) {
        this.element = document.getElementById(containerId);
        if (!this.element) {
            console.warn('Connection status container not found');
            return;
        }
        this.render();
    },

    setStatus(status) {
        if (this.status === status) return;

        this.status = status;
        if (status === 'connected') {
            this.lastUpdate = new Date();
        }
        this.render();
    },

    updateLastSeen() {
        this.lastUpdate = new Date();
        this.render();
    },

    render() {
        if (!this.element) return;

        const statusText = {
            connected: 'Connected',
            connecting: 'Connecting...',
            disconnected: 'Disconnected'
        };

        const timeAgo = this.lastUpdate ? this.getTimeAgo(this.lastUpdate) : '';
        const subtitle = this.status === 'connected' && timeAgo ?
            `<span class="connection-time">Updated ${timeAgo}</span>` : '';

        this.element.className = `connection-status ${this.status}`;
        this.element.innerHTML = `
            <span class="connection-dot"></span>
            <span class="connection-text">${statusText[this.status]}</span>
            ${subtitle}
        `;
    },

    getTimeAgo(date) {
        const seconds = Math.floor((new Date() - date) / 1000);

        if (seconds < 5) return 'just now';
        if (seconds < 60) return `${seconds}s ago`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    }
};

// ========================================
// Image Loading Handler
// ========================================
const ImageLoader = {
    loadWithFallback(img, src, options = {}) {
        const {
            timeout = 10000,
            onLoad = null,
            onError = null,
            retryCount = 2,
            retryDelay = 1000
        } = options;

        let attempts = 0;
        let timeoutId = null;

        const cleanup = () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
                timeoutId = null;
            }
        };

        const tryLoad = () => {
            attempts++;
            img.classList.add('loading');

            // Set timeout
            timeoutId = setTimeout(() => {
                if (attempts <= retryCount) {
                    console.warn(`Image load timeout, retry ${attempts}/${retryCount}`);
                    tryLoad();
                } else {
                    img.classList.remove('loading');
                    if (onError) onError(new Error('Image load timeout'));
                }
            }, timeout);

            // Update src with cache buster
            const separator = src.includes('?') ? '&' : '?';
            img.src = `${src}${separator}t=${Date.now()}`;
        };

        img.onload = () => {
            cleanup();
            img.classList.remove('loading');
            if (onLoad) onLoad();
        };

        img.onerror = () => {
            cleanup();
            if (attempts < retryCount) {
                setTimeout(tryLoad, retryDelay);
            } else {
                img.classList.remove('loading');
                if (onError) onError(new Error('Image load failed'));
            }
        };

        tryLoad();
    },

    createSkeleton(container) {
        const skeleton = document.createElement('div');
        skeleton.className = 'skeleton skeleton-image';
        container.appendChild(skeleton);
        return skeleton;
    },

    createOfflinePlaceholder() {
        return `
            <div class="offline-placeholder">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <rect x="2" y="3" width="20" height="14" rx="2"/>
                    <circle cx="8" cy="10" r="2"/>
                    <path d="M14 8h4M14 12h4"/>
                    <path d="M2 17l20 0"/>
                    <path d="M6 21l12 0"/>
                </svg>
                <span>Feed Offline</span>
            </div>
        `;
    },

    createErrorPlaceholder(message = 'Failed to load') {
        return `
            <div class="offline-placeholder">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="1.5">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M12 8v4M12 16v.01"/>
                </svg>
                <span class="text-error">${message}</span>
            </div>
        `;
    }
};

// ========================================
// Keyboard Navigation
// ========================================
const KeyboardNav = {
    init(gridSelector, cardSelector) {
        const grid = document.querySelector(gridSelector);
        if (!grid) return;

        grid.setAttribute('role', 'grid');

        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            const cards = Array.from(document.querySelectorAll(cardSelector));
            const focusedIndex = cards.findIndex(card => card === document.activeElement);

            let newIndex = -1;
            const columns = Math.floor(grid.clientWidth / cards[0]?.clientWidth) || 4;

            switch (e.key) {
                case 'ArrowRight':
                    if (focusedIndex < cards.length - 1) newIndex = focusedIndex + 1;
                    break;
                case 'ArrowLeft':
                    if (focusedIndex > 0) newIndex = focusedIndex - 1;
                    break;
                case 'ArrowDown':
                    if (focusedIndex + columns < cards.length) newIndex = focusedIndex + columns;
                    break;
                case 'ArrowUp':
                    if (focusedIndex - columns >= 0) newIndex = focusedIndex - columns;
                    break;
                case 'Home':
                    newIndex = 0;
                    break;
                case 'End':
                    newIndex = cards.length - 1;
                    break;
            }

            if (newIndex >= 0 && newIndex < cards.length) {
                e.preventDefault();
                cards[newIndex].focus();
            }
        });

        // Make cards focusable
        document.querySelectorAll(cardSelector).forEach((card, index) => {
            card.setAttribute('tabindex', index === 0 ? '0' : '-1');
            card.setAttribute('role', 'gridcell');
        });
    }
};

// ========================================
// Local Storage Helpers
// ========================================
const Storage = {
    prefix: 'cctv_',

    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(this.prefix + key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.warn('Storage get error:', e);
            return defaultValue;
        }
    },

    set(key, value) {
        try {
            localStorage.setItem(this.prefix + key, JSON.stringify(value));
            return true;
        } catch (e) {
            console.warn('Storage set error:', e);
            return false;
        }
    },

    remove(key) {
        try {
            localStorage.removeItem(this.prefix + key);
            return true;
        } catch (e) {
            return false;
        }
    }
};

// ========================================
// Format Utilities
// ========================================
const Format = {
    number(num) {
        return new Intl.NumberFormat().format(num);
    },

    bytes(bytes, decimals = 1) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
    },

    duration(seconds) {
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${mins}m`;
    },

    timeAgo(date) {
        const seconds = Math.floor((new Date() - new Date(date)) / 1000);

        const intervals = {
            year: 31536000,
            month: 2592000,
            week: 604800,
            day: 86400,
            hour: 3600,
            minute: 60
        };

        for (const [unit, value] of Object.entries(intervals)) {
            const interval = Math.floor(seconds / value);
            if (interval >= 1) {
                return `${interval} ${unit}${interval !== 1 ? 's' : ''} ago`;
            }
        }

        return 'just now';
    }
};

// ========================================
// Accessibility Helpers
// ========================================
const A11y = {
    announceToScreenReader(message, priority = 'polite') {
        const announcement = document.createElement('div');
        announcement.setAttribute('role', 'status');
        announcement.setAttribute('aria-live', priority);
        announcement.setAttribute('aria-atomic', 'true');
        announcement.className = 'sr-only';
        announcement.textContent = message;

        document.body.appendChild(announcement);

        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    },

    trapFocus(element) {
        const focusableElements = element.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        const handler = (e) => {
            if (e.key !== 'Tab') return;

            if (e.shiftKey && document.activeElement === firstElement) {
                e.preventDefault();
                lastElement.focus();
            } else if (!e.shiftKey && document.activeElement === lastElement) {
                e.preventDefault();
                firstElement.focus();
            }
        };

        element.addEventListener('keydown', handler);
        firstElement?.focus();

        return () => element.removeEventListener('keydown', handler);
    }
};

// ========================================
// Export for ES modules (if supported)
// ========================================
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        Toast,
        debounce,
        throttle,
        ConnectionStatus,
        ImageLoader,
        KeyboardNav,
        Storage,
        Format,
        A11y
    };
}
