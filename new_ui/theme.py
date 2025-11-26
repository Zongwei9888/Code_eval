"""
Theme Configuration for Code Eval v4.0
Cyberpunk-inspired dark theme with vibrant accents
"""

# Color Palette - Cyberpunk/Neo-Tokyo Theme
COLORS = {
    # Base colors
    "bg_primary": "#0a0a0f",
    "bg_secondary": "#12121a", 
    "bg_card": "#1a1a24",
    "bg_elevated": "#22222e",
    
    # Text colors
    "text_primary": "#f0f0f5",
    "text_secondary": "#a0a0b0",
    "text_dim": "#606070",
    
    # Accent colors - Neon
    "primary": "#00d4ff",       # Cyan neon
    "secondary": "#7c3aed",     # Violet
    "success": "#00ff88",       # Matrix green
    "warning": "#ffaa00",       # Amber
    "error": "#ff3366",         # Hot pink
    "info": "#3b82f6",          # Blue
    
    # Agent-specific colors
    "scanner": "#00d4ff",       # Cyan
    "analyzer": "#ffaa00",      # Amber
    "fixer": "#00ff88",         # Green
    "executor": "#3b82f6",      # Blue
    "reporter": "#d946ef",      # Fuchsia
    
    # Special
    "llm": "#7c3aed",           # Violet for LLM responses
    "tool": "#f97316",          # Orange for tool calls
    "code_bg": "#0d1117",       # GitHub dark code bg
    "border": "#2a2a3a",
    "glow": "#00d4ff33",
}

# Gradient definitions
GRADIENTS = {
    "primary": "linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%)",
    "success": "linear-gradient(135deg, #00ff88 0%, #00d4ff 100%)",
    "warm": "linear-gradient(135deg, #ff3366 0%, #ffaa00 100%)",
    "hero": "linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%)",
    "card": "linear-gradient(145deg, #1a1a24 0%, #12121a 100%)",
}

# CSS Styles
GLOBAL_CSS = """
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Orbitron:wght@500;700&display=swap');

/* Root variables */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #1a1a24;
    --bg-elevated: #22222e;
    --text-primary: #f0f0f5;
    --text-secondary: #a0a0b0;
    --text-dim: #606070;
    --primary: #00d4ff;
    --secondary: #7c3aed;
    --success: #00ff88;
    --warning: #ffaa00;
    --error: #ff3366;
    --border: #2a2a3a;
    --glow: rgba(0, 212, 255, 0.2);
}

/* Global styles */
body {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Ensure html element doesn't restrict width */
html {
    width: 100% !important;
    max-width: 100% !important;
}

/* Fix Quasar layout container */
#q-app, .q-layout, .q-page-container, .q-page {
    width: 100% !important;
    max-width: 100% !important;
}

/* Fix nicegui-content container */
.nicegui-content > div {
    width: 100% !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: var(--bg-secondary);
}
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}

/* Card styling */
.glass-card {
    background: rgba(26, 26, 36, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 12px;
}

/* Glow effect */
.neon-glow {
    box-shadow: 0 0 20px var(--glow), 0 0 40px rgba(0, 212, 255, 0.1);
}

/* Hover animations */
.hover-lift {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.hover-lift:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
}

/* Code font */
.mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Headings with Orbitron */
.cyber-heading {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 2px;
}

/* Status badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Agent chat bubbles */
.chat-bubble {
    padding: 12px 16px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 85%;
    position: relative;
}
.chat-bubble-left {
    border-left: 3px solid var(--primary);
    background: var(--bg-card);
}
.chat-bubble-right {
    border-right: 3px solid var(--warning);
    background: rgba(249, 115, 22, 0.1);
    margin-left: auto;
}

/* Log items */
.log-item {
    padding: 8px 12px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.log-success { background: rgba(0, 255, 136, 0.1); color: var(--success); }
.log-error { background: rgba(255, 51, 102, 0.1); color: var(--error); }
.log-warning { background: rgba(255, 170, 0, 0.1); color: var(--warning); }
.log-info { background: rgba(0, 212, 255, 0.1); color: var(--primary); }

/* File tree */
.file-tree-item {
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    transition: background 0.15s ease;
}
.file-tree-item:hover {
    background: var(--bg-elevated);
}
.file-tree-item.selected {
    background: rgba(0, 212, 255, 0.15);
    border-left: 2px solid var(--primary);
}

/* Workflow node */
.workflow-node {
    padding: 12px 20px;
    border-radius: 8px;
    border: 2px solid var(--border);
    background: var(--bg-card);
    text-align: center;
    min-width: 100px;
    transition: all 0.2s ease;
}
.workflow-node.active {
    border-color: var(--primary);
    box-shadow: 0 0 20px var(--glow);
}
.workflow-node.completed {
    border-color: var(--success);
}

/* Metric card */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    font-family: 'Orbitron', sans-serif;
}
.metric-label {
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Button styles */
.btn-primary {
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4) !important;
}

/* Tab styling */
.q-tab--active {
    color: var(--primary) !important;
}
.q-tab-panel {
    background: transparent !important;
    width: 100% !important;
}

/* Fix NiceGUI/Quasar layout issues - ensure full width */
.q-tab-panels {
    width: 100% !important;
}
.q-tab-panels > .q-tab-panel {
    width: 100% !important;
}
.q-page {
    width: 100% !important;
}
.nicegui-content {
    width: 100% !important;
    max-width: 100% !important;
}
.q-page-container {
    width: 100% !important;
}

/* Ensure flex containers expand properly */
.flex-grow {
    flex-grow: 1 !important;
    min-width: 0 !important;
}

/* Fix main content container */
main, .q-layout__page {
    width: 100% !important;
}

/* Input styling */
.q-field__control {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
.q-field--focused .q-field__control {
    border-color: var(--primary) !important;
}

/* Pulse animation */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 5px var(--primary); }
    50% { box-shadow: 0 0 20px var(--primary), 0 0 30px var(--glow); }
}
.pulse-glow {
    animation: pulse-glow 2s ease-in-out infinite;
}

/* Typing animation */
@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}
.typing {
    overflow: hidden;
    white-space: nowrap;
    animation: typing 2s steps(40, end);
}

/* Fade in */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in {
    animation: fadeIn 0.3s ease forwards;
}
"""

# Quasar config for NiceGUI
QUASAR_CONFIG = {
    "dark": True,
    "brand": {
        "primary": COLORS["primary"],
        "secondary": COLORS["secondary"],
        "accent": COLORS["success"],
        "dark": COLORS["bg_primary"],
        "positive": COLORS["success"],
        "negative": COLORS["error"],
        "info": COLORS["info"],
        "warning": COLORS["warning"],
    }
}

