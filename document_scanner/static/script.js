const fileInput = document.getElementById('file-input');
const dropzone = document.getElementById('dropzone');
const dropzoneEmpty = document.getElementById('dropzone-empty');
const previewImage = document.getElementById('preview-image');
const filePreviewState = document.getElementById('file-preview-state');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const scanForm = document.getElementById('scan-form');
const scanButton = document.getElementById('scan-button');

function formatFileSize(bytes) {
  if (!bytes) return 'Ready to scan';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isImageFile(file) {
  return Boolean(file && file.type && file.type.startsWith('image/'));
}

function showPreview(file) {
  if (!file) return;

  if (fileName) fileName.textContent = file.name;
  if (fileSize) fileSize.textContent = `${formatFileSize(file.size)} · Ready to scan`;

  if (isImageFile(file) && previewImage) {
    const reader = new FileReader();
    reader.onload = (event) => {
      previewImage.src = event.target.result;
      previewImage.hidden = false;
      if (dropzoneEmpty) dropzoneEmpty.hidden = true;
      if (filePreviewState) filePreviewState.hidden = true;
      if (dropzone) dropzone.classList.add('has-preview');
    };
    reader.readAsDataURL(file);
    return;
  }

  if (previewImage) {
    previewImage.hidden = true;
    previewImage.removeAttribute('src');
  }
  if (dropzoneEmpty) dropzoneEmpty.hidden = true;
  if (filePreviewState) filePreviewState.hidden = false;
  if (dropzone) dropzone.classList.add('has-preview');
}

if (fileInput) {
  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
      showPreview(fileInput.files[0]);
    }
  });
}

if (dropzone) {
  ['dragenter', 'dragover'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file && fileInput) {
      try {
        fileInput.files = event.dataTransfer.files;
      } catch (error) {
        console.warn('The browser did not allow assigning dropped files.', error);
      }
      showPreview(file);
    }
  });
}

if (scanForm && scanButton) {
  scanForm.addEventListener('submit', () => {
    scanButton.disabled = true;
    const buttonLabel = scanButton.querySelector('span');
    if (buttonLabel) buttonLabel.textContent = 'Preparing scan…';
    scanButton.setAttribute('aria-busy', 'true');
  });
}

const copyBtn = document.getElementById('copy-btn');
const extractedText = document.getElementById('extracted-text');

if (copyBtn && extractedText) {
  copyBtn.addEventListener('click', async () => {
    const originalMarkup = copyBtn.innerHTML;
    try {
      await navigator.clipboard.writeText(extractedText.textContent.trim());
      copyBtn.innerHTML = '<span class="copy-success">✓</span> Copied to clipboard';
      copyBtn.classList.add('is-copied');
      window.setTimeout(() => {
        copyBtn.innerHTML = originalMarkup;
        copyBtn.classList.remove('is-copied');
      }, 1700);
    } catch (error) {
      console.error('Copy failed', error);
      copyBtn.textContent = 'Copy unavailable';
      window.setTimeout(() => {
        copyBtn.innerHTML = originalMarkup;
      }, 1700);
    }
  });
}
