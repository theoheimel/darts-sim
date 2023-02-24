importScripts("https://npmcdn.com/regl/dist/regl.js");

let sliderValues = {};
let slidersChanged = true;
let regl, tick;

onmessage = (evt) => {
    Object.assign(sliderValues, evt.data.sliderValues);
    slidersChanged = true;

    if ("canvas" in evt.data) {
        regl = createREGL({
            canvas: evt.data.canvas,
            attributes: {
                antialias: true,
            },
            extensions: [
                'OES_texture_float',
                'OES_texture_float_linear',
                'ANGLE_instanced_arrays',
                'WEBGL_color_buffer_float'
            ],
            optionalExtensions: [
            ]
        });
    }

    if (!("resolution" in evt.data.sliderValues)) return;

    if (tick !== undefined) tick.cancel();
    const makeTexture = (pixels, filter='nearest') => regl.framebuffer({
        color: regl.texture({
            width: pixels,
            height: pixels,
            format: 'rgba',
            type: 'float',
            min: 'nearest',
            mag: filter,
        })
    });

    const resolution = 2**evt.data.sliderValues.resolution;
    const textureSize = 2*resolution;

    const kernel = makeTexture(textureSize, 'linear');
    const boardHighRes = makeTexture(textureSize);
    const boardLowRes = makeTexture(textureSize);
    const fftKernel = makeTexture(textureSize);
    const fftPing = makeTexture(textureSize);
    const fftPong = makeTexture(textureSize);
    const red256 = makeTexture(256);
    const red16 = makeTexture(16);
    const redKernel = makeTexture(1);
    const redOutput = makeTexture(1);
    const output = makeTexture(textureSize, 'linear');

    const configureComputePasses = regl({
        vert: `
        attribute vec2 position;
        void main() {
            gl_Position = vec4(position, 0, 1);
        }
        `,
        attributes: {
            position: regl.buffer([
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0,  1.0],
                [-1.0,  1.0],
                [1.0, -1.0],
                [1.0,  1.0]
            ])
        },
        count: 6,
        depth: {enable: false},
    })

    const initKernel = regl({
        frag: `
        precision mediump float;

        uniform mat2 uInvCovariance;
        uniform vec2 uOffset;
        uniform float uNorm;
        uniform float uResolution;
        uniform bool uHighRes;

        const int DOWNSAMPLING = 8;

        const float dartsSize = 170.;
        const float PI = 3.14159265359;

        float kernelProb(vec2 texCoords) {
            float zoomFactor = uHighRes ? 1. : 2.;
            vec2 dartsCoords = zoomFactor * (2. * texCoords / uResolution - 1.)
                             * dartsSize - uOffset;
            return exp(- 0.5 * dot(dartsCoords, uInvCovariance * dartsCoords))
                   * (zoomFactor * zoomFactor * dartsSize * dartsSize) * uNorm;

        }

        void main() {
            if (gl_FragCoord.x >= uResolution || gl_FragCoord.y >= uResolution) {
                gl_FragColor = vec4(0, 0, 0, 0);
                return;
            }

            float prob = 0.;
            for (int i = 0; i < DOWNSAMPLING; i++) {
                for (int j = 0; j < DOWNSAMPLING; j++) {
                    vec2 offset = (vec2(i,j) + 0.5) / float(DOWNSAMPLING) - 0.5;
                    prob += kernelProb(gl_FragCoord.xy + offset);
                }
            }
            //float score = dartsScore(gl_FragCoord.xy);
            gl_FragColor = vec4(prob / float(DOWNSAMPLING * DOWNSAMPLING), 0, 0, 0);
        }
        `,
        uniforms: {
            uInvCovariance: regl.prop("invCovariance"),
            uOffset: regl.prop("offset"),
            uNorm: regl.prop("norm"),
            uResolution: resolution,
            uHighRes: regl.prop("highRes")
        },
        framebuffer: regl.prop("output")
    });
    const computeCovariance = (sigma, excentricity, angle) => {
        const majorAxis = (sigma * excentricity)**2;
        const minorAxis = sigma**2;
        const sin = Math.sin(angle);
        const cos = Math.cos(angle);
        const var1 = majorAxis * cos**2 + minorAxis * sin**2;
        const var2 = majorAxis * sin**2 + minorAxis * cos**2;
        const cov12 = (majorAxis - minorAxis) * sin * cos;
        const det = majorAxis * minorAxis;
        return {
            norm: 1 / (2 * Math.PI * Math.sqrt(det)),
            covariance: [var1, cov12, cov12, var2],
            invCovariance: [var2/det, -cov12/det, -cov12/det, var1/det]
        }
    };

    const initBoard = regl({
        frag: `
        precision mediump float;

        uniform float uSegments[20];
        uniform float uResolution;

        const int DOWNSAMPLING = 8;

        const float PI = 3.14159265359;
        const float R_BULL = 12.7 / 2.0;
        const float R_DOUBLE_BULL = 31.8 / 2.0;
        const float R_TRIPLE_OUTER = 107.;
        const float R_TRIPLE_INNER = R_TRIPLE_OUTER - 8.;
        const float R_DOUBLE_OUTER = 170.;
        const float R_DOUBLE_INNER = R_DOUBLE_OUTER - 8.;

        float dartsScore(vec2 texCoords) {
            vec2 dartsCoords = (texCoords / (0.5*uResolution) - 1.) * R_DOUBLE_OUTER;
            float r = length(dartsCoords);
            float phi = atan(dartsCoords.y, dartsCoords.x);
            float score, segment;
            int segmentIdx = int(floor(mod(phi/PI*10. + 15.5, 20.)));
            for (int i = 0; i < 20; i++) {
                if (i == segmentIdx) {
                    segment = uSegments[i];
                }
            }
            if (r < R_BULL) {
                score = 50.;
            } else if (r < R_DOUBLE_BULL) {
                score = 25.;
            } else if (r < R_TRIPLE_INNER) {
                score = segment;
            } else if (r < R_TRIPLE_OUTER) {
                score = 3. * segment;
            } else if (r < R_DOUBLE_INNER) {
                score = segment;
            } else if (r < R_DOUBLE_OUTER) {
                score = 2. * segment;
            } else {
                score = 0.;
            }
            return score;
        }

        void main() {
            if (gl_FragCoord.x >= uResolution || gl_FragCoord.y >= uResolution) {
                gl_FragColor = vec4(0.,0.,0.,0.);
                return;
            }
            float score = 0.;
            for (int i = 0; i < DOWNSAMPLING; i++) {
                for (int j = 0; j < DOWNSAMPLING; j++) {
                    vec2 offset = (vec2(i,j) + 0.5) / float(DOWNSAMPLING) - 0.5;
                    score += dartsScore(gl_FragCoord.xy + offset);
                }
            }
            //float score = dartsScore(gl_FragCoord.xy);
            gl_FragColor = vec4(score / float(DOWNSAMPLING * DOWNSAMPLING), 0, 0, 0);
        }
        `,
        uniforms: {
            uSegments: [20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 6, 13, 4, 18, 1],
            uResolution: regl.prop("resolution")
        },
        framebuffer: regl.prop("output")
    });

    const multiply = regl({
        frag: `
        precision mediump float;

        uniform sampler2D uMat1;
        uniform sampler2D uMat2;
        uniform float uTextureSize;

        void main() {
            vec4 pix1 = texture2D(uMat1, gl_FragCoord.xy / uTextureSize);
            vec4 pix2 = texture2D(uMat2, gl_FragCoord.xy / uTextureSize);
            gl_FragColor = vec4(
                pix1.x*pix2.x - pix1.y*pix2.y,
                pix1.x*pix2.y + pix1.y*pix2.x,
                0,
                0
            );
        }
        `,
        framebuffer: regl.prop("output"),
        uniforms: {
            uMat1: regl.prop("mat1"),
            uMat2: regl.prop("mat2"),
            uTextureSize: textureSize
        }
    });

    /*
     * The FFT code is based on the glsl-fft project by Ricky Reusser
     *   https://github.com/rreusser/glsl-fft
     * and his blog post
     *   https://observablehq.com/@rreusser/simulating-the-1-d-schrodinger-equation-in-webgl
     *
     */

    const performFFTPasses = regl({
        frag: `
        precision highp float;

        uniform sampler2D uSrc;
        uniform vec2 uResolution;
        uniform float uSubtransformSize, uNormalization;
        uniform bool uHorizontal, uForward;

        const float TWOPI = 6.283185307179586;

        vec4 fft (
            sampler2D src,
            vec2 resolution,
            float subtransformSize,
            bool horizontal,
            bool forward,
            float normalization
        ) {
            vec2 evenPos, oddPos, twiddle, outputA, outputB;
            vec4 even, odd;
            float index, evenIndex, twiddleArgument;

            index = (horizontal ? gl_FragCoord.x : gl_FragCoord.y) - 0.5;

            evenIndex = floor(index / subtransformSize) *
                (subtransformSize * 0.5) +
                mod(index, subtransformSize * 0.5) +
                0.5;

            if (horizontal) {
                evenPos = vec2(evenIndex, gl_FragCoord.y);
                oddPos = vec2(evenIndex, gl_FragCoord.y);
            } else {
                evenPos = vec2(gl_FragCoord.x, evenIndex);
                oddPos = vec2(gl_FragCoord.x, evenIndex);
            }

            evenPos *= resolution;
            oddPos *= resolution;

            if (horizontal) {
                oddPos.x += 0.5;
            } else {
                oddPos.y += 0.5;
            }

            even = texture2D(src, evenPos);
            odd = texture2D(src, oddPos);

            twiddleArgument = (forward ? TWOPI : -TWOPI) * (index / subtransformSize);
            twiddle = vec2(cos(twiddleArgument), sin(twiddleArgument));

            return (even.rgba + vec4(
                twiddle.x * odd.xz - twiddle.y * odd.yw,
                twiddle.y * odd.xz + twiddle.x * odd.yw
            ).xzyw) * normalization;
        }

        void main () {
            gl_FragColor = fft(
                uSrc, uResolution, uSubtransformSize, uHorizontal, uForward, uNormalization
            );
        }`,
        uniforms: {
            uSrc: regl.prop('input'),
            uResolution: regl.prop('resolution'),
            uSubtransformSize: regl.prop('subtransformSize'),
            uHorizontal: regl.prop('horizontal'),
            uForward: regl.prop('forward'),
            uNormalization: regl.prop('normalization'),
        },
        framebuffer: regl.prop('output'),
    });

    function planFFT (opts) {
        function isPowerOfTwo(n) {
            return n !== 0 && (n & (n - 1)) === 0
        }

        function checkPOT (label, value) {
            if (!isPowerOfTwo(value)) {
                throw new Error(label + ' must be a power of two. got ' + label + ' = ' + value);
            }
        }
        var i, ping, pong, uniforms, tmp, width, height;

        opts = opts || {};
        opts.forward = opts.forward === undefined ? true : opts.forward;
        opts.splitNormalization = opts.splitNormalization === undefined ? true : opts.splitNormalization;

        function swap () {
            tmp = ping;
            ping = pong;
            pong = tmp;
        }

        if (opts.size !== undefined) {
            width = height = opts.size;
            checkPOT('size', width);
        } else if (opts.width !== undefined && opts.height !== undefined) {
            width = opts.width;
            height = opts.height;
            checkPOT('width', width);
            checkPOT('height', width);
        } else {
            throw new Error('either size or both width and height must provided.');
        }

        // Swap to avoid collisions with the input:
        ping = opts.ping;
        if (opts.input === opts.pong) {
            ping = opts.pong;
        }
        pong = ping === opts.ping ? opts.pong : opts.ping;

        var passes = [];
        var xIterations = Math.round(Math.log(width) / Math.log(2));
        var yIterations = Math.round(Math.log(height) / Math.log(2));
        var iterations = xIterations + yIterations;

        // Swap to avoid collisions with output:
        if (opts.output === ((iterations % 2 === 0) ? pong : ping)) {
            swap();
        }

        // If we've avoiding collision with output creates an input collision,
        // then you'll just have to rework your framebuffers and try again.
        if (opts.input === pong) {
            throw new Error([
                'not enough framebuffers to compute without copying data. You may perform',
                'the computation with only two framebuffers, but the output must equal',
                'the input when an even number of iterations are required.'
            ].join(' '));
        }

        for (i = 0; i < iterations; i++) {
            uniforms = {
                input: ping,
                output: pong,
                horizontal: i < xIterations,
                forward: !!opts.forward,
                resolution: [1.0 / width, 1.0 / height]
            };

            if (i === 0) {
                uniforms.input = opts.input;
            } else if (i === iterations - 1) {
                uniforms.output = opts.output;
            }

            if (i === 0) {
                if (!!opts.splitNormalization) {
                    uniforms.normalization = 1.0 / Math.sqrt(width * height);
                } else if (!opts.forward) {
                    uniforms.normalization = 1.0 / width / height;
                } else {
                    uniforms.normalization = 1;
                }
            } else {
                uniforms.normalization = 1;
            }

            uniforms.subtransformSize = Math.pow(2, (uniforms.horizontal ? i : (i - xIterations)) + 1);

            passes.push(uniforms);

            swap();
        }

        passes[0].input = opts.input;
        passes[passes.length - 1].output = opts.output;

        return passes;
    }

    let forwardBoardHighResFFT = planFFT({
        width: textureSize,
        height: textureSize,
        input: boardHighRes,
        ping: fftPing,
        pong: fftPong,
        output: boardHighRes,
        forward: true
    });
    let forwardBoardLowResFFT = planFFT({
        width: textureSize,
        height: textureSize,
        input: boardLowRes,
        ping: fftPing,
        pong: fftPong,
        output: boardLowRes,
        forward: true
    });
    let forwardKernelFFT = planFFT({
        width: textureSize,
        height: textureSize,
        input: kernel,
        ping: fftPing,
        pong: fftPong,
        output: fftKernel,
        forward: true
    });
    let inverseOutputFFT = planFFT({
        width: textureSize,
        height: textureSize,
        input: output,
        ping: fftPing,
        pong: fftPong,
        output: output,
        forward: false
    });

    const makeReduce = (reductionFactor) => regl({
        frag: `
        precision mediump float;

        uniform sampler2D uInput;
        uniform float uOffset, uInputSize;
        uniform bool uFirstPass;
        const int reductionFactor = ${reductionFactor};

        void main() {
            float maxVal = -1.e9;
            float minVal = 1.e9;
            float sumVal = 0.;
            for (int i = 0; i < reductionFactor; i++) {
                for (int j = 0; j < reductionFactor; j++) {
                    vec2 subPixel = vec2(i,j);
                    vec2 coord = (
                        uOffset + (gl_FragCoord.xy - 0.5) * float(reductionFactor)
                        + subPixel + 0.5
                    ) / uInputSize;
                    vec4 value = texture2D(uInput, coord);
                    if (uFirstPass) {
                        maxVal = max(maxVal, value.x);
                        minVal = min(minVal, value.x);
                        sumVal = sumVal + value.x;
                    } else {
                        maxVal = max(maxVal, value.x);
                        minVal = min(minVal, value.y);
                        sumVal = sumVal + value.z;
                    }
                }
            }
            gl_FragColor = vec4(
                maxVal,
                minVal,
                sumVal,
                0
            );
        }
        `,
        framebuffer: regl.prop("output"),
        uniforms: {
            uInput: regl.prop("input"),
            uOffset: regl.prop("offset"),
            uInputSize: regl.prop("inputSize"),
            uFirstPass: regl.prop("firstPass"),
        }
    });
    const reduceOps = {
        16: makeReduce(16),
        8: makeReduce(8),
        4: makeReduce(4), 
        2: makeReduce(2)
    };
    const reduce = (opt) => {
        if (opt.size > 256) {
            reduceOps[opt.size / 256]({
                input: opt.input,
                output: red256,
                offset: opt.offset,
                inputSize: opt.totalSize,
                firstPass: true
            });
            reduceOps[16]({
                input: red256,
                output: red16,
                offset: 0,
                inputSize: 256,
                firstPass: false
            });
        } else {
            reduceOps[opt.size / 16]({
                input: opt.input,
                output: red16,
                offset: opt.offset,
                inputSize: opt.totalSize,
                firstPass: true
            });

        }
        reduceOps[16]({
            input: red16,
            output: opt.output,
            offset: 0,
            inputSize: 16,
            firstPass: false
        });
    };

    const configureViewport = regl({
        blend: {
            enable: true,
            func: {
                srcRGB: 'src alpha',
                srcAlpha: 1,
                dstRGB: 'one minus src alpha',
                dstAlpha: 1,
            },
            equation: {
                rgb: 'add',
                alpha: 'add',
            }
        },
        depth: {enable: false},
    })

    const drawTexture = regl({
        vert: `
        attribute vec2 position;
        void main() {
            gl_Position = vec4(position, 0, 1);
        }
        `,

        frag: `
        precision mediump float;
        uniform sampler2D uLeft;
        uniform sampler2D uRight;
        uniform sampler2D uRedLeft;
        uniform sampler2D uRedRight;
        uniform bool uHighRes;
        uniform float uWidth, uHeight;

        // Copyright 2019 Google LLC.
        // SPDX-License-Identifier: Apache-2.0

        // Polynomial approximation in GLSL for the Turbo colormap
        // Original LUT: https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f

        // Authors:
        //   Colormap Design: Anton Mikhailov (mikhailov@google.com)
        //   GLSL Approximation: Ruofei Du (ruofei@google.com)
        vec4 cmap(float x) {
            const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
            const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
            const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
            const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
            const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
            const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);

            x = clamp(x,0.0,1.0);
            vec4 v4 = vec4( 1.0, x, x * x, x * x * x);
            vec2 v2 = v4.zw * v4.z;
            return vec4(
                dot(v4, kRedVec4)   + dot(v2, kRedVec2),
                dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
                dot(v4, kBlueVec4)  + dot(v2, kBlueVec2),
                1.
            );
        }

        void main() {
            vec4 red, value;
            if (gl_FragCoord.x < uWidth/2.) {
                vec2 coord = vec2(
                    gl_FragCoord.x / uWidth * 2.,
                    gl_FragCoord.y / uHeight
                ) * (uHighRes ? 0.5 : 0.25) + 0.25;
                red = texture2D(uRedLeft, vec2(0.5, 0.5));
                value = texture2D(uLeft, coord);
            } else {
                vec2 coord = vec2(
                    gl_FragCoord.x / uWidth * 2. - 1.,
                    gl_FragCoord.y / uHeight
                );
                if (uHighRes) {
                    coord = coord * 0.5;
                } else {
                    coord = coord * 0.25 + 0.125;
                }
                red = texture2D(uRedRight, vec2(0.5, 0.5));
                value = texture2D(uRight, coord);
            }
            float redMax = red.x;
            float redMin = 0.; //red.y;
            gl_FragColor = cmap((value.x - redMin) / (redMax - redMin));
            //gl_FragColor = cmap(redMax);
            //gl_FragColor = cmap(texture2D(uInput, coord).x/512./60.);
            //gl_FragColor = cmap(texture2D(uInput, gl_FragCoord.xy / 1024.).x/60.);
            //gl_FragColor = vec4(texture2D(uInput, gl_FragCoord.xy / 1024.).xy, 0., 1.);
        }
        `,

        attributes: {
            position: regl.buffer([
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0,  1.0],
                [-1.0,  1.0],
                [1.0, -1.0],
                [1.0,  1.0]
            ])
        },
        count: 6,
        uniforms: {
            uRight: regl.prop("right"),
            uLeft: regl.prop("left"),
            uRedRight: regl.prop("redRight"),
            uRedLeft: regl.prop("redLeft"),
            uHighRes: regl.prop("highRes"),
            uWidth: ctx => ctx.framebufferWidth,
            uHeight: ctx => ctx.framebufferHeight,
        }
    });

    const render = (firstRender) => {
        regl.clear({color: [0, 0, 0, 0], depth: 1});
        const cov = computeCovariance(
            sliderValues.sigma,
            sliderValues.excentricity,
            sliderValues.angle / 180. * Math.PI
        );
        const highRes = sliderValues.sigma * sliderValues.excentricity < 42.;
        configureComputePasses(() => {
            if (firstRender) {
                initBoard({output: boardHighRes, resolution: resolution});
                initBoard({output: boardLowRes, resolution: resolution/2});
                performFFTPasses(forwardBoardHighResFFT);
                performFFTPasses(forwardBoardLowResFFT);
            }
            initKernel({
                output: kernel,
                invCovariance: cov.invCovariance,
                norm: cov.norm,
                offset: [
                    sliderValues.xoff * sliderValues.sigma,
                    sliderValues.yoff * sliderValues.sigma
                ],
                highRes: highRes
            });
            reduce({
                input: kernel,
                offset: 0,
                size: resolution,
                totalSize: textureSize,
                output: redKernel
            });
            performFFTPasses(forwardKernelFFT);
            multiply({
                mat1: highRes ? boardHighRes : boardLowRes,
                mat2: fftKernel,
                output: output
            });
            performFFTPasses(inverseOutputFFT);
            reduce({
                input: output,
                offset: resolution/2,
                size: resolution,
                totalSize: textureSize,
                output: redOutput
            });
        });
        configureViewport(() => drawTexture({
            left: output,
            right: kernel,
            redLeft: redOutput,
            redRight: redKernel,
            highRes: highRes
        }));
    }
    render(true);

    tick = regl.frame(() => {
        if (!slidersChanged) return;
        slidersChanged = false;
        render(false);
    });
}
