async function submitVerification(mode, onProgress) {
    let response;

    if (mode === 'image') {
        const file = document.getElementById('img-upload')?.files?.[0];
        if (!file) throw new Error('Select an image before analyzing it.');
        const body = new FormData();
        body.append('image', file);
        response = await fetch('/api/v1/verify/start', { method: 'POST', body });
    } else if (mode === 'url') {
        const url = document.getElementById('url-input')?.value.trim();
        response = await fetch('/api/v1/verify/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'url', url })
        });
    } else {
        response = await fetch('/api/v1/verify/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'text', claim: getCurrentClaimText() })
        });
    }

    const started = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(started.error || 'Could not start verification.');
    if (!started.job_id) throw new Error('Backend did not return a verification job ID.');

    // Poll the independent job endpoint while LangGraph runs in its worker.
    let lastSequence = 0;
    for (;;) {
        await new Promise(resolve => setTimeout(resolve, 650));
        const progressResponse = await fetch(
            '/api/v1/verify/progress/' + encodeURIComponent(started.job_id),
            { cache: 'no-store' }
        );
        const job = await progressResponse.json().catch(() => ({}));
        if (!progressResponse.ok) {
            throw new Error(job.error || 'Could not read verification progress.');
        }

        const events = Array.isArray(job.events) ? job.events : [];
        for (const event of events) {
            const sequence = Number(event.sequence) || 0;
            if (sequence > lastSequence) {
                lastSequence = sequence;
                if (typeof onProgress === 'function') onProgress(event);
            }
        }

        if (job.status === 'completed') return job.result;
        if (job.status === 'failed') {
            throw new Error(job.error || 'Verification failed in the backend.');
        }
    }
}
