const Vec = {
    dist: (p1, p2) => Math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2),
    distSqr: (p1, p2) => (p1.x - p2.x)**2 + (p1.y - p2.y)**2,
    add: (p1, p2) => ({ x: p1.x + p2.x, y: p1.y + p2.y }),
    sub: (p1, p2) => ({ x: p1.x - p2.x, y: p1.y - p2.y }),
    mid: (p1, p2) => ({ x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 }),
    scale: (p, s) => ({ x: p.x * s, y: p.y * s }),
};

/**
 * Utility for matrix operations, specifically solving the 8x8 system for homography.
 */
const Matrix = {
    /**
     * Solves the system Ax = b using Gaussian elimination.
     * A is a 2D array (matrix), b is a 1D array (vector).
     */
    gaussianElimination: (A, b) => {
        const n = A.length;
        for (let i = 0; i < n; i++) {
            // Find pivot
            let maxEl = Math.abs(A[i][i]);
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(A[k][i]) > maxEl) {
                    maxEl = Math.abs(A[k][i]);
                    maxRow = k;
                }
            }

            // Swap rows
            [A[i], A[maxRow]] = [A[maxRow], A[i]];
            [b[i], b[maxRow]] = [b[maxRow], b[i]];

            // Handle singularity
            if (Math.abs(A[i][i]) < 1e-10) {
                console.error("Matrix is singular");
                throw new Error("Matrix is singular");
            }

            // Make pivot 1
            const pivot = A[i][i];
            for (let k = i; k < n; k++) {
                A[i][k] /= pivot;
            }
            b[i] /= pivot;

            // Eliminate other rows
            for (let k = 0; k < n; k++) {
                if (k !== i) {
                    const factor = A[k][i];
                    for (let j = i; j < n; j++) {
                        A[k][j] -= factor * A[i][j];
                    }
                    b[k] -= factor * b[i];
                }
            }
        }
        return b; // b now contains the solution vector x
    },

    /**
     * Multiplies a 3x3 matrix by a 3x1 vector (point in homogeneous coords).
     */
    multiply: (m, v) => {
        return [
            m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
            m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
            m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
        ];
    },

    /**
     * Inverts a 3x3 matrix. Needed for inverse mapping.
     */
    invert: (m) => {
        const [m0, m1, m2, m3, m4, m5, m6, m7, m8] = m;
        const det = m0 * (m4 * m8 - m5 * m7) -
                    m1 * (m3 * m8 - m5 * m6) +
                    m2 * (m3 * m7 - m4 * m6);
        
        if (Math.abs(det) < 1e-10) return null; // Not invertible
        
        const invDet = 1 / det;
        
        return [
            (m4 * m8 - m5 * m7) * invDet,
            (m2 * m7 - m1 * m8) * invDet,
            (m1 * m5 - m2 * m4) * invDet,
            (m5 * m6 - m3 * m8) * invDet,
            (m0 * m8 - m2 * m6) * invDet,
            (m2 * m3 - m0 * m5) * invDet,
            (m3 * m7 - m4 * m6) * invDet,
            (m1 * m6 - m0 * m7) * invDet,
            (m0 * m4 - m1 * m3) * invDet,
        ];
    }
};

/**
 * The core Computer Vision module.
 * Contains from-scratch implementations of Canny, Hough, Homography, and Transform.
 */
const CV = {
    /**
     * Converts an ImageData object to grayscale.
     */
    grayscale: (imgData) => {
        const data = imgData.data;
        const grayData = new Uint8ClampedArray(imgData.width * imgData.height);
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            // Using NTSC/PAL luminance formula
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;
            grayData[i / 4] = gray;
        }
        return { data: grayData, width: imgData.width, height: imgData.height };
    },

    /**
     * Applies Gaussian Blur using a simple 3x3 kernel.
     */
    gaussianBlur: (grayImg) => {
        const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
        const kernelWeight = 16;
        const { data, width, height } = grayImg;
        const blurredData = new Uint8ClampedArray(data.length);

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                let sum = 0;
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const idx = (y + ky) * width + (x + kx);
                        const kernelIdx = (ky + 1) * 3 + (kx + 1);
                        sum += data[idx] * kernel[kernelIdx];
                    }
                }
                blurredData[y * width + x] = sum / kernelWeight;
            }
        }
        return { data: blurredData, width, height };
    },

    /**
     * Applies Sobel operator to find intensity gradients.
     */
    sobel: (grayImg) => {
        const { data, width, height } = grayImg;
        const magnitudes = new Float32Array(data.length);
        const angles = new Float32Array(data.length);
        const Gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
        const Gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                let sumX = 0;
                let sumY = 0;
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const idx = (y + ky) * width + (x + kx);
                        const kernelIdx = (ky + 1) * 3 + (kx + 1);
                        sumX += data[idx] * Gx[kernelIdx];
                        sumY += data[idx] * Gy[kernelIdx];
                    }
                }
                const idx = y * width + x;
                magnitudes[idx] = Math.sqrt(sumX**2 + sumY**2);
                angles[idx] = Math.atan2(sumY, sumX);
            }
        }
        return { magnitudes, angles, width, height };
    },

    /**
     * Non-maximum suppression for Canny edge detection.
     */
    nonMaxSuppression: (sobelData) => {
        const { magnitudes, angles, width, height } = sobelData;
        const nmsData = new Float32Array(magnitudes.length);

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                const mag = magnitudes[idx];
                const ang = angles[idx] * 180 / Math.PI;

                let q = 255;
                let r = 255;

                // Normalize angle
                let angle = ang < 0 ? ang + 180 : ang;

                if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)) {
                    // Horizontal edge
                    q = magnitudes[idx + 1];
                    r = magnitudes[idx - 1];
                } else if (22.5 <= angle && angle < 67.5) {
                    // +45 degree edge
                    q = magnitudes[idx + width + 1];
                    r = magnitudes[idx - width - 1];
                } else if (67.5 <= angle && angle < 112.5) {
                    // Vertical edge
                    q = magnitudes[idx + width];
                    r = magnitudes[idx - width];
                } else if (112.5 <= angle && angle < 157.5) {
                    // -45 degree edge
                    q = magnitudes[idx - width + 1];
                    r = magnitudes[idx + width - 1];
                }

                if (mag >= q && mag >= r) {
                    nmsData[idx] = mag;
                }
            }
        }
        return { data: nmsData, width, height };
    },

    /**
     * Hysteresis thresholding for Canny.
     */
    hysteresis: (nmsData) => {
        const { data, width, height } = nmsData;
        const edgeData = new Uint8ClampedArray(data.length);
        // NOTE: These thresholds are sensitive. 70/30 is a good general guess.
        const HIGH_THRESHOLD = 70;
        const LOW_THRESHOLD = 30;
        
        const STRONG = 255;
        const WEAK = 100;

        // Double thresholding
        for (let i = 0; i < data.length; i++) {
            if (data[i] > HIGH_THRESHOLD) edgeData[i] = STRONG;
            else if (data[i] > LOW_THRESHOLD) edgeData[i] = WEAK;
        }

        // Edge tracking
        const trace = (i) => {
            const x = i % width;
            const y = Math.floor(i / width);
            if (x < 0 || x >= width || y < 0 || y >= height) return;
            
            if (edgeData[i] === WEAK) {
                edgeData[i] = STRONG;
                trace(i - width - 1); trace(i - width); trace(i - width + 1);
                trace(i - 1);                 trace(i + 1);
                trace(i + width - 1); trace(i + width); trace(i + width + 1);
            }
        };
        
        for (let i = 0; i < data.length; i++) {
            if (edgeData[i] === STRONG) trace(i);
        }

        // Final cleanup
        for (let i = 0; i < data.length; i++) {
            if (edgeData[i] !== STRONG) edgeData[i] = 0;
        }
        
        return { data: edgeData, width, height };
    },

    /**
     * Canny edge detection pipeline.
     */
    canny: (imgData) => {
        const gray = CV.grayscale(imgData);
        const blurred = CV.gaussianBlur(gray);
        const sobel = CV.sobel(blurred);
        const nms = CV.nonMaxSuppression(sobel);
        const edges = CV.hysteresis(nms);
        return edges;
    },

    /**
     * Hough Transform to detect lines in the edge image.
     */
    houghTransform: (edgeImg, numLinesToFind = 20) => {
        const { data, width, height } = edgeImg;
        
        // Parameter space
        const maxRho = Math.sqrt(width**2 + height**2);
        const numThetas = 180; // 1 degree resolution
        const numRhos = Math.floor(maxRho * 2); // from -maxRho to +maxRho
        
        // Accumulator array
        const accumulator = new Uint32Array(numThetas * numRhos);
        
        const anSin = [];
        const anCos = [];
        for (let t = 0; t < numThetas; t++) {
            const theta = t * Math.PI / 180;
            anSin[t] = Math.sin(theta);
            anCos[t] = Math.cos(theta);
        }

        // Voting
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                if (data[y * width + x] === 255) { // If it's an edge pixel
                    for (let t = 0; t < numThetas; t++) {
                        const rho = Math.round(x * anCos[t] + y * anSin[t]);
                        const rhoIdx = rho + Math.floor(maxRho); // Offset for negative rho
                        accumulator[rhoIdx * numThetas + t]++;
                    }
                }
            }
        }

        // Find peaks in accumulator
        const peaks = [];
        // Use a lower vote threshold to get more candidates
        const VOTE_THRESHOLD = Math.min(width, height) * 0.2; // 20% of smallest dim
        for (let r = 0; r < numRhos; r++) {
            for (let t = 0; t < numThetas; t++) {
                const votes = accumulator[r * numThetas + t];
                if (votes > VOTE_THRESHOLD) { 
                    peaks.push({ votes, rho: r - Math.floor(maxRho), theta: t });
                }
            }
        }
        
        // Sort by votes and take top N
        peaks.sort((a, b) => b.votes - a.votes);
        
        // Non-maximal suppression in (rho, theta) space
        const lines = [];
        const RHO_THRESH = 20;
        const THETA_THRESH = 10;
        
        for (const peak of peaks) {
            if (lines.length >= numLinesToFind) break;
            let isUnique = true;
            for (const line of lines) {
                if (Math.abs(peak.rho - line.rho) < RHO_THRESH && Math.abs(peak.theta - line.theta) < THETA_THRESH) {
                    isUnique = false;
                    break;
                }
            }
            if (isUnique) {
                lines.push(peak);
            }
        }
        
        return lines; // [{ rho, theta }, ...]
    },

    /**
     * Solves the 3x3 homography matrix H given 4 source and 4 destination points.
     */
    solveHomography: (src, dst) => {
        // We need to solve Ah = b, where h is the 8 unknown parameters of H
        // h = [h11, h12, h13, h21, h22, h23, h31, h32] (h33 is 1)
        const A = [];
        const b = [];

        for (let i = 0; i < 4; i++) {
            const { x: x, y: y } = src[i];
            const { x: xp, y: yp } = dst[i];
            
            A.push([x, y, 1, 0, 0, 0, -x*xp, -y*xp]);
            b.push(xp);
            A.push([0, 0, 0, x, y, 1, -x*yp, -y*yp]);
            b.push(yp);
        }

        try {
            const h = Matrix.gaussianElimination(A, b);
            // Add h33 = 1
            return [...h, 1]; // [h11, h12, h13, h21, h22, h23, h31, h32, 1]
        } catch(e) {
            console.error("Failed to solve homography:", e);
            return null;
        }
    },

    /**
     * Transforms the source image to the destination using the inverse homography.
     * Uses Bilinear Interpolation for high quality.
     */
    transformImage: (srcImgData, invH, dstWidth, dstHeight) => {
        const dstImgData = new ImageData(dstWidth, dstHeight);
        const src = srcImgData.data;
        const dst = dstImgData.data;
        const srcW = srcImgData.width;
        const srcH = srcImgData.height;

        for (let y = 0; y < dstHeight; y++) {
            for (let x = 0; x < dstWidth; x++) {
                // 1. Find corresponding point in source image
                // [wx', wy', w] = H_inv * [x, y, 1]
                const [wx_p, wy_p, w] = Matrix.multiply(invH, [x, y, 1]);
                
                // Convert from homogeneous to 2D
                const srcX = wx_p / w;
                const srcY = wy_p / w;

                // 2. Check if it's outside the source image bounds
                if (srcX < 0 || srcX > srcW - 2 || srcY < 0 || srcY > srcH - 2) {
                    continue; // Leave pixel as transparent black
                }

                // 3. Perform Bilinear Interpolation
                const x1 = Math.floor(srcX);
                const y1 = Math.floor(srcY);
                const x2 = x1 + 1;
                const y2 = y1 + 1;
                
                const dx = srcX - x1;
                const dy = srcY - y1;
                
                const idx = (y * dstWidth + x) * 4;

                for (let c = 0; c < 3; c++) { // For R, G, B
                    const c11 = src[(y1 * srcW + x1) * 4 + c];
                    const c21 = src[(y1 * srcW + x2) * 4 + c];
                    const c12 = src[(y2 * srcW + x1) * 4 + c];
                    const c22 = src[(y2 * srcW + x2) * 4 + c];
                    
                    const top = c11 * (1 - dx) + c21 * dx;
                    const bottom = c12 * (1 - dx) + c22 * dx;
                    const val = top * (1 - dy) + bottom * dy;
                    
                    dst[idx + c] = val;
                }
                dst[idx + 3] = 255; // Alpha
            }
        }
        return dstImgData;
    }
};


/**
 * Main Application Class
 * Manages state, UI, and event listeners.
 */
class PerspectiveApp {
    constructor() {
        // DOM Elements
        this.canvas = document.getElementById('mainCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.loupeCanvas = document.getElementById('loupeCanvas');
        this.loupeCtx = this.loupeCanvas.getContext('2d');
        this.outputCanvas = document.getElementById('outputCanvas');
        this.outputCtx = this.outputCanvas.getContext('2d');
        
        this.cameraInput = document.getElementById('cameraInput');
        
        this.autoCorrectButton = document.getElementById('autoCorrectButton');
        this.transformButton = document.getElementById('transformButton');
        this.resetButtonBefore = document.getElementById('resetButtonBefore');
        this.backButton = document.getElementById('backButton');
        this.downloadButton = document.getElementById('downloadButton');
        
        this.mainControls = document.getElementById('mainControls');
        this.outputControls = document.getElementById('outputControls');
        this.loader = document.getElementById('loader');
        this.loaderText = document.getElementById('loaderText');

        // State
        this.originalImage = null; // The full-resolution image element
        this.originalImageData = null; // Full-res pixel data
        this.scaledImageData = null; // Downscaled data for CV
        this.scaleFactor = 1; // Factor used for downscaling
        
        this.imagePos = { x: 0, y: 0, scale: 1 };
        this.quadPoints = []; // 4 corner points in *image* coordinates
        this.draggedHandle = null; // index (0-3)
        this.touchState = {
            panning: false,
            pinching: false,
            lastPos: { x: 0, y: 0 },
            lastDist: 0,
        };
        
        this.HANDLE_SIZE = 15; // In logical pixels
        this.LOUPE_ZOOM = 15;

        this.bindEvents();
        this.resize();
        requestAnimationFrame(this.render.bind(this));
    }

    bindEvents() {
        window.addEventListener('resize', this.resize.bind(this));
        
        this.cameraInput.addEventListener('click', this.loadImage.bind(this));
        
        this.canvas.addEventListener('touchstart', this.onTouchStart.bind(this), { passive: false });
        this.canvas.addEventListener('touchmove', this.onTouchMove.bind(this), { passive: false });
        this.canvas.addEventListener('touchend', this.onTouchEnd.bind(this), { passive: false });

        // Button Listeners
        this.autoCorrectButton.addEventListener('click', this.onAutoCorrect.bind(this));
        this.transformButton.addEventListener('click', this.onTransform.bind(this));
        this.resetButtonBefore.addEventListener('click', this.reset.bind(this));
        this.backButton.addEventListener('click', this.showMainView.bind(this));
        this.downloadButton.addEventListener('click', this.shareOrDownloadImage.bind(this));
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    showLoader(text) {
        this.loaderText.textContent = text;
        this.loader.style.display = 'flex';
    }

    hideLoader() {
        this.loader.style.display = 'none';
    }

    // This function handles input from camera
    loadImage(e) {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            this.originalImage = new Image();
            this.originalImage.onload = () => {
                this.showLoader('Processing image...');
                // Use a timeout to allow the loader to render
                setTimeout(() => {
                    try {
                        this.processImage();
                    } catch (err) {
                        console.error("Error processing image:", err);
                        this.showModal("Error processing image. It might be too large.");
                    }
                    this.hideLoader();
                }, 50);
            };
            this.originalImage.onerror = () => {
                this.showModal("Failed to load image file.");
                this.hideLoader();
            };
            this.originalImage.src = event.target.result;
        };
        reader.readAsDataURL(file);
        
        // Reset the input value. This is crucial to allow
        // taking the same picture or loading the same file twice.
        e.target.value = null;
    }

    processImage() {
        const { width, height } = this.originalImage;
        
        // --- 1. Store full-res image data for final transform ---
        const tempCanvas = (typeof OffscreenCanvas !== "undefined")
            ? new OffscreenCanvas(width, height)
            : document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        
        const tempCtx = tempCanvas.getContext('2d');
        if (!tempCtx) {
            throw new Error("Could not create 2D context for image processing.");
        }
        tempCtx.drawImage(this.originalImage, 0, 0);
        this.originalImageData = tempCtx.getImageData(0, 0, width, height);

        // --- 2. Create scaled-down version for CV operations ---
        const MAX_CV_DIM = 800; // Performance constraint
        this.scaleFactor = 1;
        let cvWidth = width, cvHeight = height;

        if (width > MAX_CV_DIM || height > MAX_CV_DIM) {
            if (width > height) {
                this.scaleFactor = width / MAX_CV_DIM;
                cvWidth = MAX_CV_DIM;
                cvHeight = Math.round(height / this.scaleFactor);
            } else {
                this.scaleFactor = height / MAX_CV_DIM;
                cvHeight = MAX_CV_DIM;
                cvWidth = Math.round(width / this.scaleFactor);
            }
        }
        
        const cvCanvas = (typeof OffscreenCanvas !== "undefined")
            ? new OffscreenCanvas(cvWidth, cvHeight)
            : document.createElement('canvas');
        cvCanvas.width = cvWidth;
        cvCanvas.height = cvHeight;
        const cvCtx = cvCanvas.getContext('2d');
        cvCtx.drawImage(this.originalImage, 0, 0, cvWidth, cvHeight);
        this.scaledImageData = cvCtx.getImageData(0, 0, cvWidth, cvHeight);
        
        // --- 3. Reset view and quad ---
        this.reset();
        this.autoCorrectButton.disabled = false;
        this.resetButtonBefore.disabled = false;
    }

    reset() {
        if (!this.originalImage) return;

        // Reset view
        const { width, height } = this.originalImage;
        const scaleX = this.canvas.width / width;
        const controlsHeight = document.getElementById('controls').offsetHeight || 160;
        const availableHeight = this.canvas.height - controlsHeight;
        const scaleY = availableHeight / height;
        
        this.imagePos.scale = Math.min(scaleX, scaleY) * 0.9; // Zoom out 10%
        this.imagePos.x = (this.canvas.width - width * this.imagePos.scale) / 2;
        this.imagePos.y = (availableHeight - height * this.imagePos.scale) / 2; // Center in top area

        // Reset quad points
        const margin = 0.2;
        this.quadPoints = [
            { x: width * margin, y: height * margin },
            { x: width * (1 - margin), y: height * margin },
            { x: width * (1 - margin), y: height * (1 - margin) },
            { x: width * margin, y: height * (1 - margin) },
        ];
        
        this.transformButton.disabled = false;
    }

    // --- Coordinate Conversion ---
    screenToImageCoords(p) {
        return {
            x: (p.x - this.imagePos.x) / this.imagePos.scale,
            y: (p.y - this.imagePos.y) / this.imagePos.scale,
        };
    }
    
    imageToScreenCoords(p) {
        return {
            x: p.x * this.imagePos.scale + this.imagePos.x,
            y: p.y * this.imagePos.scale + this.imagePos.y,
        };
    }

    // --- Touch Event Handlers ---
    
    onTouchStart(e) {
        e.preventDefault();
        const touches = e.touches;
        
        if (!this.originalImage) return;

        if (touches.length === 1) {
            const touch = touches[0];
            const touchPos = { x: touch.clientX, y: touch.clientY };
            this.touchState.lastPos = touchPos;
            
            // Check if touching a handle
            const imgTouchPos = this.screenToImageCoords(touchPos);
            const handleRadius = this.HANDLE_SIZE / this.imagePos.scale;
            
            this.draggedHandle = null;
            let minD = Infinity;
            
            for (let i = 0; i < this.quadPoints.length; i++) {
                const d = Vec.distSqr(imgTouchPos, this.quadPoints[i]);
                if (d < handleRadius**2 && d < minD) {
                    minD = d;
                    this.draggedHandle = i;
                }
            }

            if (this.draggedHandle !== null) {
                this.touchState.panning = false;
                this.showLoupe(touchPos);
            } else {
                // Start panning
                this.touchState.panning = true;
            }
        } else if (touches.length === 2) {
            // Start pinching
            this.touchState.pinching = true;
            this.touchState.panning = false;
            this.draggedHandle = null;
            this.hideLoupe();
            this.touchState.lastDist = Vec.dist(touches[0], touches[1]);
            this.touchState.lastPos = Vec.mid(touches[0], touches[1]);
        }
    }

    onTouchMove(e) {
        e.preventDefault();
        const touches = e.touches;

        if (this.draggedHandle !== null && touches.length === 1) {
            // Dragging a handle
            const touchPos = { x: touches[0].clientX, y: touches[0].clientY };
            this.quadPoints[this.draggedHandle] = this.screenToImageCoords(touchPos);
            this.updateLoupe(touchPos);

        } else if (this.touchState.panning && touches.length === 1) {
            // Panning
            const touchPos = { x: touches[0].clientX, y: touches[0].clientY };
            const delta = Vec.sub(touchPos, this.touchState.lastPos);
            this.imagePos.x += delta.x;
            this.imagePos.y += delta.y;
            this.touchState.lastPos = touchPos;

        } else if (this.touchState.pinching && touches.length === 2) {
            // Pinch-zooming
            const touch1 = { x: touches[0].clientX, y: touches[0].clientY };
            const touch2 = { x: touches[1].clientX, y: touches[1].clientY };
            
            const newDist = Vec.dist(touch1, touch2);
            const newMid = Vec.mid(touch1, touch2);
            
            const scaleChange = newDist / this.touchState.lastDist;
            
            // Get world coords of midpoint
            const worldPos = this.screenToImageCoords(newMid);
            
            // Update scale
            this.imagePos.scale *= scaleChange;
            
            // Get new screen coords of world point
            const newScreenPos = this.imageToScreenCoords(worldPos);
            
            // Adjust imagePos to keep world point at midpoint
            this.imagePos.x -= (newScreenPos.x - newMid.x);
            this.imagePos.y -= (newScreenPos.y - newMid.y);
            
            this.touchState.lastDist = newDist;
            this.touchState.lastPos = newMid;
        }
    }
    
    onTouchEnd(e) {
        e.preventDefault();
        this.draggedHandle = null;
        this.touchState.panning = false;
        this.touchState.pinching = false;
        this.hideLoupe();
    }

    // --- Loupe Methods ---
    
    showLoupe(pos) {
        this.loupeCanvas.style.display = 'block';
        this.updateLoupe(pos);
    }

    hideLoupe() {
        this.loupeCanvas.style.display = 'none';
    }

    updateLoupe(pos) {
        // Position loupe above finger
        this.loupeCanvas.style.left = `${pos.x - 60}px`; // 60 = half width
        this.loupeCanvas.style.top = `${pos.y - 160}px`; // 160 = height + offset
        
        const imgPos = this.screenToImageCoords(pos);
        
        const loupeW = this.loupeCanvas.width;
        const loupeH = this.loupeCanvas.height;
        
        const srcW = loupeW / this.LOUPE_ZOOM;
        const srcH = loupeH / this.LOUPE_ZOOM;
        
        const srcX = imgPos.x - srcW / 2;
        const srcY = imgPos.y - srcH / 2;
        
        this.loupeCtx.imageSmoothingEnabled = false; // Pixelated zoom
        this.loupeCtx.clearRect(0, 0, loupeW, loupeH);
        this.loupeCtx.drawImage(
            this.originalImage,
            srcX, srcY, srcW, srcH, // Source rect
            0, 0, loupeW, loupeH   // Dest rect
        );
    }

    // --- Main Render Loop ---
    
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (this.originalImage) {
            // Apply pan/zoom
            this.ctx.save();
            this.ctx.translate(this.imagePos.x, this.imagePos.y);
            this.ctx.scale(this.imagePos.scale, this.imagePos.scale);
            
            // Draw the image
            this.ctx.drawImage(this.originalImage, 0, 0);

            // Only draw the quad if it has been initialized
            if (this.quadPoints.length === 4) {
                // Draw the quad
                this.ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
                this.ctx.lineWidth = 2 / this.imagePos.scale;
                this.ctx.beginPath();
                this.ctx.moveTo(this.quadPoints[0].x, this.quadPoints[0].y);
                for (let i = 1; i <= 4; i++) {
                    this.ctx.lineTo(this.quadPoints[i % 4].x, this.quadPoints[i % 4].y);
                }
                this.ctx.stroke();

                // Draw handles
                this.ctx.fillStyle = 'rgba(255, 0, 0, 0.6)';
                const handleRadius = this.HANDLE_SIZE / this.imagePos.scale;
                for (let i = 0; i < 4; i++) {
                    this.ctx.beginPath();
                    this.ctx.arc(this.quadPoints[i].x, this.quadPoints[i].y, handleRadius, 0, Math.PI * 2);
                    this.ctx.fill();
                }
            }
            
            this.ctx.restore();
        } else {
            // Prompt to load image
            this.ctx.fillStyle = '#999';
            this.ctx.font = '18px sans-serif'; // Fallback font
            this.ctx.textAlign = 'center';
            this.ctx.fillText('Tap the camera icon to begin.', this.canvas.width / 2, this.canvas.height / 2);
        }
        
        requestAnimationFrame(this.render.bind(this));
    }

    // --- Action Button Handlers ---

    async onAutoCorrect() {
        if (!this.scaledImageData) return;

        this.showLoader('Running Canny Edge Detection...');
        
        // Run CV pipeline in chunks to allow UI to update
        await new Promise(r => setTimeout(r, 50));
        const edges = CV.canny(this.scaledImageData);
        
        this.showLoader('Running Hough Transform...');
        await new Promise(r => setTimeout(r, 50));
        const lines = CV.houghTransform(edges, 40); // Find top 40 lines
        
        if (lines.length < 4) {
            this.hideLoader();
            this.showModal("Auto-Correct failed: Not enough strong lines found.");
            return;
        }

        this.showLoader('Matching lines...');
        await new Promise(r => setTimeout(r, 50));
        
        // User's quad lines (in scaled CV coordinates)
        const userLines = [];
        for (let i = 0; i < 4; i++) {
            const p1 = this.quadPoints[i];
            const p2 = this.quadPoints[(i + 1) % 4];
            
            // Get midpoint in full-res image coords
            const mid = Vec.mid(p1, p2);
            
            // Convert to scaled CV coords
            const cvMid = Vec.scale(mid, 1 / this.scaleFactor);
            
            // Get angle
            let theta = Math.atan2(p2.y - p1.y, p2.x - p1.x); // Angle of line
            theta += Math.PI / 2; // Angle of normal
            
            // Normalize theta to 0-pi
            while (theta < 0) theta += Math.PI;
            while (theta >= Math.PI) theta -= Math.PI;
            
            // Get rho
            const rho = cvMid.x * Math.cos(theta) + cvMid.y * Math.sin(theta);
            
            userLines.push({ rho, theta: theta * 180 / Math.PI }); // theta in degrees
        }
        
        // Find the 4 best-matching lines from the Hough transform
        const matchedLines = [];
        const usedHoughIndices = new Set();

        for (const userLine of userLines) {
            let bestMatch = null;
            let minCost = Infinity;

            for (let i = 0; i < lines.length; i++) {
                if (usedHoughIndices.has(i)) continue;
                
                const houghLine = lines[i]; // { rho, theta (degrees) }
                
                // Cost function: penalize angle difference heavily
                const angleDiff = Math.abs(userLine.theta - houghLine.theta);
                const angleCost = Math.min(angleDiff, 180 - angleDiff) * 5; // High weight
                const rhoCost = Math.abs(userLine.rho - houghLine.rho);
                
                const cost = angleCost + rhoCost;

                if (cost < minCost) {
                    minCost = cost;
                    bestMatch = i;
                }
            }

            if (bestMatch !== null) {
                matchedLines.push(lines[bestMatch]);
                usedHoughIndices.add(bestMatch);
            }
        }
        
        if (matchedLines.length < 4) {
            this.hideLoader();
            this.showModal("Auto-Correct failed: Could not find 4 distinct lines.");
            return;
        }
        
        // --- Find intersections of the 4 matched lines ---
        const getIntersection = (line1, line2) => {
            const t1 = line1.theta * Math.PI / 180;
            const t2 = line2.theta * Math.PI / 180;
            const r1 = line1.rho;
            const r2 = line2.rho;
            
            const cos1 = Math.cos(t1), sin1 = Math.sin(t1);
            const cos2 = Math.cos(t2), sin2 = Math.sin(t2);
            
            const det = cos1 * sin2 - sin1 * cos2;
            if (Math.abs(det) < 1e-6) return null; // Parallel lines
            
            const x = (r1 * sin2 - r2 * sin1) / det;
            const y = (r2 * cos1 - r1 * cos2) / det;
            
            return { x, y };
        };
        
        const newPoints = [
            getIntersection(matchedLines[3], matchedLines[0]),
            getIntersection(matchedLines[0], matchedLines[1]),
            getIntersection(matchedLines[1], matchedLines[2]),
            getIntersection(matchedLines[2], matchedLines[3]),
        ];

        // Check if all intersections are valid
        if (newPoints.some(p => p === null)) {
            this.hideLoader();
            this.showModal("Auto-Correct failed: Matched lines are parallel.");
            return;
        }
        
        // Update quad points
        this.quadPoints = newPoints.map(p => Vec.scale(p, this.scaleFactor));
        this.hideLoader();
    }

    async onTransform() {
        if (!this.originalImageData) return;

        this.showLoader('Solving Homography...');
        await new Promise(r => setTimeout(r, 50));

        const srcPoints = this.quadPoints;
        
        // Define destination rect. We'll average the lengths of
        // opposite sides to get a "natural" aspect ratio.
        const w1 = Vec.dist(srcPoints[0], srcPoints[1]);
        const w2 = Vec.dist(srcPoints[2], srcPoints[3]);
        const h1 = Vec.dist(srcPoints[1], srcPoints[2]);
        const h2 = Vec.dist(srcPoints[0], srcPoints[3]);
        
        const dstW = Math.round((w1 + w2) / 2);
        const dstH = Math.round((h1 + h2) / 2);
        
        if (dstW <= 0 || dstH <= 0) {
             this.hideLoader();
             this.showModal("Transform failed: Invalid dimensions. Please check quad points.");
             return;
        }

        const dstPoints = [
            { x: 0, y: 0 },
            { x: dstW, y: 0 },
            { x: dstW, y: dstH },
            { x: 0, y: dstH },
        ];

        const H = CV.solveHomography(srcPoints, dstPoints);
        if (!H) {
            this.hideLoader();
            this.showModal("Transform failed: Could not solve matrix. Check points (e.g., are 3 collinear?)");
            return;
        }
        
        const H_inv = Matrix.invert(H);
        if (!H_inv) {
            this.hideLoader();
            this.showModal("Transform failed: Matrix is not invertible.");
            return;
        }
        
        this.showLoader('Applying transform... (this can take a while)');
        await new Promise(r => setTimeout(r, 50));
        
        const outputImgData = CV.transformImage(this.originalImageData, H_inv, dstW, dstH);
        
        this.outputCanvas.width = dstW;
        this.outputCanvas.height = dstH;
        this.outputCtx.putImageData(outputImgData, 0, 0);

        this.showOutputView();
        this.hideLoader();
    }
    
    showMainView() {
        this.outputControls.style.display = 'none';
        this.mainControls.style.display = 'flex';

        // Show the main canvas and hide the output canvas
        this.outputCanvas.style.display = 'none';
        this.canvas.style.display = 'block';

        this.reset();
    }
    
    showOutputView() {
        this.mainControls.style.display = 'none';
        this.outputControls.style.display = 'flex';
        // Hide the main canvas (original image) and show the output (transformed) canvas
        this.canvas.style.display = 'none';
        this.outputCanvas.style.display = 'block';
    }
    
    /**
     * Tries to use the Web Share API. Falls back to programmatic download.
     */
    async shareOrDownloadImage() {
        // 1. Get the blob from the canvas
        const blob = await new Promise(resolve => this.outputCanvas.toBlob(resolve, 'image/png'));
        
        if (!blob) {
            this.showModal("Error creating image file.");
            return;
        }

        const file = new File([blob], 'transformed_image.png', { type: 'image/png' });
        const shareData = {
            files: [file],
            title: 'Transformed Image',
            text: 'Here is the image I corrected.',
        };

        // 2. Try Web Share API first (for mobile)
        if (navigator.share && navigator.canShare && navigator.canShare(shareData)) {
            try {
                await navigator.share(shareData);
                // Share successful
            } catch (err) {
                if (err.name !== 'AbortError') {
                    console.error('Share failed:', err);
                    // If share fails, fall back to download
                    this.triggerDownload(file); 
                }
                // If it IS AbortError, the user just cancelled. Do nothing.
            }
        } else {
            // 3. Fallback to programmatic download
            this.triggerDownload(file);
        }
    }

    /**
     * Helper function to trigger a file download programmatically.
     */
    triggerDownload(file) {
        try {
            // Use Object URL for the file, which is cleaner than dataURL
            const url = URL.createObjectURL(file);
            const link = document.createElement('a');
            link.href = url;
            link.download = file.name;
            
            // Append to body (required for Firefox)
            document.body.appendChild(link);
            
            link.click();
            
            // Clean up
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error("Download failed:", err);
            this.showModal("Download failed. Your browser might be blocking it.");
        }
    }

    // Custom modal to avoid using alert()
    showModal(message) {
        let modal = document.getElementById('alertModal');
        if (modal) modal.remove();
        
        modal = document.createElement('div');
        modal.id = 'alertModal';
        // Using inline styles from the generated CSS
        modal.style = "position:absolute; top:20px; left:50%; transform:translateX(-50%); background:rgba(255, 69, 58, 0.9); color:white; padding: 12px 20px; border-radius: 8px; z-index: 2000; font-weight: 500; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);";
        modal.innerHTML = `<p>${message}</p>`;
        
        document.body.appendChild(modal);
        
        setTimeout(() => {
            if (modal) {
                modal.style.transition = 'opacity 0.5s';
                modal.style.opacity = '0';
                setTimeout(() => modal.remove(), 500);
            }
        }, 3000);
    }
}

// --- Initialize the App ---
document.addEventListener('DOMContentLoaded', () => {
    const app = new PerspectiveApp();
});