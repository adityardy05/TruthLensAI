/* ============================================================
   TruthLens · js/export.js
   Real PDF export via jsPDF (loaded from CDN in index.html).

   The report content comes from REPORT_* in data/mock-data.js plus
   whatever is on screen, so the PDF and the Results view can never
   drift apart. If jsPDF is unavailable (offline), the button says so
   instead of failing silently.
   ============================================================ */

function exportReport() {
    const btn = document.getElementById('btn-export');
    const text = document.getElementById('export-text');
    const icon = document.getElementById('export-icon');
    if (!btn || !text || !icon) return;

    // Close the dropdown (it opens on focus-within)
    if (document.activeElement) document.activeElement.blur();

    if (!window.jspdf || !window.jspdf.jsPDF) {
        text.innerText = 'Export unavailable (offline)';
        icon.innerText = 'error';
        setTimeout(() => {
            text.innerText = 'Download Full Report';
            icon.innerText = 'download';
        }, 3000);
        return;
    }

    text.innerText = 'Exporting...';
    icon.innerText = 'hourglass_empty';
    btn.classList.add('opacity-70', 'pointer-events-none');

    // The existing PDF drop animation, kept as visual feedback
    const pdfIcon = document.getElementById('mock-pdf');
    if (pdfIcon) {
        pdfIcon.classList.remove('hidden', 'mock-pdf-anim');
        void pdfIcon.offsetWidth;
        pdfIcon.classList.add('mock-pdf-anim');
        setTimeout(() => {
            pdfIcon.classList.add('hidden');
            pdfIcon.classList.remove('mock-pdf-anim');
        }, 2000);
    }

    setTimeout(() => {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        // The shared current claim first, then history, then the raw input
        const history = getHistory();
        const claim = (hasCurrentClaim() && getCurrentClaim().text)
            || (history[0] && history[0].claim)
            || getCurrentClaimText()
            || 'Claim not available';
        const result = getCurrentClaim()?.result;
        const verdict = result?.verdict || readVerdictFromResults();

        // Header
        doc.setFontSize(20);
        doc.setFont('helvetica', 'bold');
        doc.text('TruthLens Verification Report', 20, 22);

        doc.setFontSize(9);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(120);
        doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 30);
        doc.text(REPORT_META.strapline, 20, 36);

        doc.setDrawColor(220);
        doc.line(20, 41, 190, 41);

        // Claim
        doc.setTextColor(0);
        doc.setFontSize(11);
        doc.setFont('helvetica', 'bold');
        doc.text('Claim Analyzed', 20, 51);
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);
        const claimLines = doc.splitTextToSize(claim, 170);
        doc.text(claimLines, 20, 59);

        let y = 59 + claimLines.length * 6 + 6;

        // A long claim can push later sections past the bottom of the
        // page; start a new page instead of silently clipping them.
        // `need` is the offset of the block's LAST text baseline from y.
        const PAGE_BOTTOM = 278;
        const ensureSpace = need => {
            if (y + need > PAGE_BOTTOM) { doc.addPage(); y = 22; }
        };

        ensureSpace(0);
        doc.line(20, y, 190, y);
        y += 8;

        // Verdict
        ensureSpace(32);
        doc.setFontSize(11);
        doc.setFont('helvetica', 'bold');
        doc.text('Verdict', 20, y);
        y += 8;
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);
        const language = (document.getElementById('lang-text')?.innerText || 'English')
            .replace(/^[^A-Za-z]+/, '')
            .replace(/ detected.*$/, '');
        doc.text(`Result:       ${verdict}`, 20, y); y += 6;
        doc.text(`Confidence:   ${Math.round(result?.confidence || 0)}%`, 20, y); y += 6;
        doc.text(`Language:     ${result?.detected_language || language}`, 20, y); y += 6;
        doc.text(`Sources:      ${result?.evidence?.length || 0} retrieved`, 20, y); y += 6;
        doc.text(`Sub-claims:   ${result?.sub_claims?.length || 0} decomposed`, 20, y); y += 8;

        doc.line(20, y, 190, y); y += 8;

        // Sub-claims
        ensureSpace(20);
        doc.setFontSize(11);
        doc.setFont('helvetica', 'bold');
        doc.text('Sub-Claim Breakdown', 20, y); y += 8;
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);
        (result?.sub_claims || []).map((label, index) => [String(index + 1).padStart(2, '0'), label, verdict]).forEach(([num, label, sub]) => {
            ensureSpace(5);
            doc.text(`${num}  ${label}`, 20, y); y += 5;
            doc.setTextColor(150);
            doc.text(`     -> ${sub}`, 20, y); y += 7;
            doc.setTextColor(0);
        });

        doc.line(20, y, 190, y); y += 8;

        // Sources
        ensureSpace(20);
        doc.setFontSize(11);
        doc.setFont('helvetica', 'bold');
        doc.text('Sources', 20, y); y += 8;
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);
        (result?.evidence || []).map(item => [item.source_domain, Number(item.combined_reliability || 0).toFixed(2), item.credibility_source || 'unrated', 'evidence']).forEach(([domain, score, rating, rel]) => {
            ensureSpace(0);
            doc.text(`- ${domain}   Score: ${score}   ${rating}   - ${rel}`, 20, y); y += 7;
        });

        doc.line(20, y, 190, y); y += 8;

        // Reviewer findings
        ensureSpace(20);
        doc.setFontSize(11);
        doc.setFont('helvetica', 'bold');
        doc.text('Reviewer Findings', 20, y); y += 8;
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);
        Object.entries(result?.persona_insights || {}).map(([name, finding]) => [name.replace(/_/g, ' '), String(finding)]).forEach(([name, finding]) => {
            const lines = doc.splitTextToSize(finding, 162);
            ensureSpace(6 + (lines.length - 1) * 5);
            doc.setFont('helvetica', 'bold');
            doc.text(name + ':', 20, y); y += 6;
            doc.setFont('helvetica', 'normal');
            doc.text(lines, 25, y); y += lines.length * 5 + 4;
        });

        // Footer
        doc.setFontSize(8);
        doc.setTextColor(160);
        doc.text(REPORT_META.footer, 20, 284);

        doc.save(`TruthLens_Report_${Date.now()}.pdf`);

        text.innerText = '✓ Downloaded';
        icon.innerText = 'check';
        btn.classList.remove('opacity-70');
        btn.classList.add('bg-tertiary-fixed/20', 'text-tertiary-container');

        setTimeout(() => {
            text.innerText = 'Download Full Report';
            icon.innerText = 'download';
            btn.classList.remove('bg-tertiary-fixed/20', 'text-tertiary-container', 'pointer-events-none');
        }, 3000);
    }, 800);
}

// Kept so any existing reference to the old stub still works
function downloadPDF() {
    exportReport();
}
