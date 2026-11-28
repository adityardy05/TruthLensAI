async function submitVerification(mode, onProgress = () => {}) {
    let body;
    let headers;

    if (mode === 'image') {
        const file = document.getElementById('img-upload')?.files?.[0];
        if (!file) throw new Error('Select an image before analyzing it.');
        body = new FormData();
        body.append('mode', 'image');
        body.append('image', file);
    } else if (mode === 'url') {
        const url = document.getElementById('url-input')?.value.trim();
        body = JSON.stringify({ mode: 'url', url });
        headers = { 'Content-Type': 'application/json' };
    } else {
        body = JSON.stringify({ mode: 'text', claim: getCurrentClaimText() });
        headers = { 'Content-Type': 'application/json' };
    }

    const response = await fetch('/api/v1/verify/stream', {
        method: 'POST',
        headers,
        body
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || 'Verification could not be started.');
    }
    if (!response.body) throw new Error('This browser does not support streamed verification updates.');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let result = null;

    const handleEvent = (block) => {
        let eventName = 'message';
        const dataLines = [];
        block.split(/\\r?\\n/).forEach(line => {
            if (line.startsWith('event:')) eventName = line.slice(6).trim();
            else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
        });
        if (!dataLines.length) return;
        const payload = JSON.parse(dataLines.join('\\n'));
        if (eventName === 'progress') onProgress(payload);
        else if (eventName === 'result') result = payload.data;
        else if (eventName === 'error') throw new Error(payload.message || 'Verification failed.');
    };

    try {
        while (true) {
            const { value, done } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
            const blocks = buffer.split(/\\r?\\n\\r?\\n/);
            buffer = blocks.pop() || '';
            blocks.forEach(handleEvent);
            if (done) break;
        }
        if (buffer.trim()) handleEvent(buffer);
    } finally {
        reader.releaseLock();
    }

    if (!result) throw new Error('The verification stream ended without a result.');
    return result;
}
