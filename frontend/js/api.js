async function submitVerification(mode) {
    let response;
    if (mode === 'image') {
        const file = document.getElementById('img-upload')?.files?.[0];
        if (!file) throw new Error('Select an image before analyzing it.');
        const body = new FormData();
        body.append('image', file);
        response = await fetch('/api/v1/verify/image', { method: 'POST', body });
    } else if (mode === 'url') {
        const url = document.getElementById('url-input')?.value.trim();
        response = await fetch('/api/v1/verify/url', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url })
        });
    } else {
        response = await fetch('/api/v1/verify', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ claim: getCurrentClaimText() })
        });
    }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || 'Verification failed.');
    return data;
}
