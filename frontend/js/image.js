/* ============================================================
   TruthLens · js/image.js
   Image upload: 5 MB validation, preview inside the upload
    placeholder, the server OCR box, and remove/replace.

   IMPORTANT: nothing in this file — and nothing in analyzeClaim() —
   may call #img-upload.click(). The dashed drop-zone in index.html
   is the only thing allowed to open the OS file picker. Clicking
   Analyze must never reopen it.
   ============================================================ */

function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (file.size > 5 * 1024 * 1024) {
        alert('Image must be under 5MB.');
        e.target.value = '';
        return;
    }

    const reader = new FileReader();
    reader.onload = ev => {
        const thumb = document.getElementById('img-thumb');
        thumb.src = ev.target.result;
        thumb.classList.remove('hidden');
        document.getElementById('img-placeholder').classList.add('hidden');
        document.getElementById('img-name').textContent = file.name;
        document.getElementById('img-name').classList.remove('hidden');
        document.getElementById('img-preview').classList.remove('hidden');

        const ocr = document.getElementById('ocr-text');
        if (ocr) ocr.value = '';
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    document.getElementById('img-preview').classList.add('hidden');
    document.getElementById('img-upload').value = '';

    const thumb = document.getElementById('img-thumb');
    thumb.removeAttribute('src');
    thumb.classList.add('hidden');

    const nameEl = document.getElementById('img-name');
    nameEl.textContent = '';
    nameEl.classList.add('hidden');

    document.getElementById('img-placeholder').classList.remove('hidden');
}
