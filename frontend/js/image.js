/* TruthLens image upload and real backend OCR preview. */
function getOcrStatusElement() {
    const ocr = document.getElementById('ocr-text');
    if (!ocr) return null;
    let status = document.getElementById('ocr-status');
    if (!status) {
        status = document.createElement('p');
        status.id = 'ocr-status';
        status.className = 'text-xs text-text-muted mb-2';
        ocr.parentElement.insertBefore(status, ocr);
    }
    return status;
}

async function handleImageUpload(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    if (!['image/png', 'image/jpeg'].includes(file.type)) {
        alert('Please select a PNG or JPG image.');
        e.target.value = '';
        return;
    }
    if (file.size > 5 * 1024 * 1024) {
        alert('Image must be under 5MB.');
        e.target.value = '';
        return;
    }

    const thumb = document.getElementById('img-thumb');
    const placeholder = document.getElementById('img-placeholder');
    const nameEl = document.getElementById('img-name');
    const preview = document.getElementById('img-preview');
    const ocr = document.getElementById('ocr-text');
    const status = getOcrStatusElement();

    if (thumb.src.startsWith('blob:')) URL.revokeObjectURL(thumb.src);
    thumb.src = URL.createObjectURL(file);
    thumb.classList.remove('hidden');
    placeholder.classList.add('hidden');
    nameEl.textContent = file.name;
    nameEl.classList.remove('hidden');
    preview.classList.remove('hidden');

    if (ocr) {
        ocr.value = '';
        ocr.placeholder = 'Extracting text from image...';
    }
    if (status) status.textContent = 'Uploading image for OCR...';

    try {
        const body = new FormData();
        body.append('image', file);
        const response = await fetch('/api/v1/ocr', { method: 'POST', body });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || 'OCR extraction failed.');
        if (!data.original_text || !data.original_text.trim()) {
            throw new Error('No readable text was found in this image.');
        }
        if (ocr) {
            ocr.value = data.original_text;
            ocr.placeholder = 'Extracted text will appear here after OCR.';
        }
        if (status) {
            status.textContent = 'OCR completed · ' +
                String(data.detected_language || 'unknown').toUpperCase() + ' detected';
        }
    } catch (error) {
        if (ocr) {
            ocr.value = '';
            ocr.placeholder = 'OCR extraction failed. Try a clearer image.';
        }
        if (status) status.textContent = error.message || 'OCR extraction failed.';
        console.error('[TruthLens OCR]', error);
    }
}

function removeImage() {
    document.getElementById('img-preview').classList.add('hidden');
    document.getElementById('img-upload').value = '';
    const thumb = document.getElementById('img-thumb');
    if (thumb.src.startsWith('blob:')) URL.revokeObjectURL(thumb.src);
    thumb.removeAttribute('src');
    thumb.classList.add('hidden');
    const nameEl = document.getElementById('img-name');
    nameEl.textContent = '';
    nameEl.classList.add('hidden');
    const ocr = document.getElementById('ocr-text');
    if (ocr) {
        ocr.value = '';
        ocr.placeholder = 'Extracted text will appear here after OCR.';
    }
    const status = document.getElementById('ocr-status');
    if (status) status.textContent = '';
    document.getElementById('img-placeholder').classList.remove('hidden');
}
