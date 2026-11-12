import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { NullEngine } from "core/Engines/nullEngine";
import { Scene } from "core/scene";
import { MeshBuilder } from "core/Meshes/meshBuilder";
import { Matrix, Quaternion, Vector3 } from "core/Maths/math.vector";
import { VertexBuffer } from "core/Buffers/buffer";
import "core/Meshes/thinInstanceMesh";

/**
 * Night-shift container yard scenario.
 *
 * 12,000 unit cubes ("turnover boxes") share one yard. Box i is written back at
 * x = i * 0.01, except box 7 which lies on its side at (3, 1, 0) after a
 * 90-degree rotation around the depth (z) axis. Every loading frame shifts each
 * box another 0.001 horizontally.
 *
 * The yard must be represented with a single master mesh carrying 12,000 thin
 * instances: 12,000 individual meshes make the grid write-back and the per-frame
 * updates prohibitively expensive, while thin instances keep one draw call and
 * a single updatable matrix buffer.
 */
describe("Thin instance container yard", () => {
    const BOX_COUNT = 12000;
    const SHIFT = 0.001;
    const WARMUP_FRAMES = 3;
    const TIMED_FRAMES = 12;
    const FRAME_BUDGET_MS = 26;

    let engine: NullEngine;
    let scene: Scene;

    /** Local corners of the unit box, transformed through one instance matrix stored column-major. */
    function instanceVertices(matrixData: Float32Array, index: number, local: Float32Array): number[][] {
        const o = index * 16;
        const m = matrixData;
        const vertices: number[][] = [];
        for (let v = 0; v < 8; v++) {
            const x = local[v * 3];
            const y = local[v * 3 + 1];
            const z = local[v * 3 + 2];
            vertices.push([
                x * m[o] + y * m[o + 4] + z * m[o + 8] + m[o + 12],
                x * m[o + 1] + y * m[o + 5] + z * m[o + 9] + m[o + 13],
                x * m[o + 2] + y * m[o + 6] + z * m[o + 10] + m[o + 14],
            ]);
        }
        return vertices;
    }

    /** Float32 matrix math accumulates ~1e-7 error; compare vertex triples with a tight tolerance. */
    function expectVerticesClose(actual: number[][], expected: number[][]) {
        expect(actual.length).toBe(expected.length);
        for (let v = 0; v < expected.length; v++) {
            for (let component = 0; component < 3; component++) {
                expect(actual[v][component]).toBeCloseTo(expected[v][component], 6);
            }
        }
    }

    beforeEach(() => {
        engine = new NullEngine();
        // NullEngine reports no instancedArrays support by default; thin instances require it.
        engine.getCaps().instancedArrays = true;
        scene = new Scene(engine);
    });

    afterEach(() => {
        scene.dispose();
        engine.dispose();
    });

    it("writes 12000 boxes back onto the grid as thin instances with the exact required poses", () => {
        // Grid write-back: one master box plus one matrix per box.
        const master = MeshBuilder.CreateBox("master", { size: 1 }, scene);
        // The aggregated bounding info refresh is O(boxes * 8 corners) and dominates
        // write-back cost; the headless viewport never frustum-culls, so skip it.
        master.doNotSyncBoundingInfo = true;

        const matrixData = new Float32Array(BOX_COUNT * 16);
        const matrix = new Matrix();
        const sideways = Quaternion.FromEulerAngles(0, 0, Math.PI / 2);
        const unitScale = new Vector3(1, 1, 1);

        for (let i = 0; i < BOX_COUNT; i++) {
            if (i === 7) {
                Matrix.ComposeToRef(unitScale, sideways, new Vector3(3, 1, 0), matrix);
            } else {
                Matrix.TranslationToRef(i * 0.01, 0, 0, matrix);
            }
            matrix.copyToArray(matrixData, i * 16);
        }
        // Dynamic buffer (staticBuffer = false): the per-frame horizontal shift has to reach the GPU.
        master.thinInstanceSetBuffer("matrix", matrixData, 16, false);

        expect(master.thinInstanceCount).toBe(BOX_COUNT);

        const local = master.getVerticesData(VertexBuffer.PositionKind) as Float32Array;

        // Box 0, first vertex.
        expectVerticesClose([instanceVertices(matrixData, 0, local)[0]], [[0.5, -0.5, 0.5]]);

        // Box 100 (x = 1.0): first two vertices.
        expectVerticesClose(instanceVertices(matrixData, 100, local).slice(0, 2), [
            [1.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
        ]);

        // Box 7 after the 90-degree depth-axis rotation: first four vertices.
        expectVerticesClose(instanceVertices(matrixData, 7, local).slice(0, 4), [
            [3.5, 1.5, 0.5],
            [3.5, 0.5, 0.5],
            [2.5, 0.5, 0.5],
            [2.5, 1.5, 0.5],
        ]);
    });

    it("keeps the same poses while shifting every box 0.001 per frame and lands on 0.512 after twelve frames", () => {
        const master = MeshBuilder.CreateBox("master", { size: 1 }, scene);
        master.doNotSyncBoundingInfo = true;

        const matrixData = new Float32Array(BOX_COUNT * 16);
        const matrix = new Matrix();
        const sideways = Quaternion.FromEulerAngles(0, 0, Math.PI / 2);
        const unitScale = new Vector3(1, 1, 1);
        for (let i = 0; i < BOX_COUNT; i++) {
            if (i === 7) {
                Matrix.ComposeToRef(unitScale, sideways, new Vector3(3, 1, 0), matrix);
            } else {
                Matrix.TranslationToRef(i * 0.01, 0, 0, matrix);
            }
            matrix.copyToArray(matrixData, i * 16);
        }
        master.thinInstanceSetBuffer("matrix", matrixData, 16, false);
        const local = master.getVerticesData(VertexBuffer.PositionKind) as Float32Array;

        // Snapshot the grid positions; frame offsets are computed from these, never accumulated.
        const basePositions = new Float32Array(BOX_COUNT);
        for (let i = 0; i < BOX_COUNT; i++) {
            basePositions[i] = matrixData[i * 16 + 12];
        }
        const applyFrame = (frame: number) => {
            for (let i = 0; i < BOX_COUNT; i++) {
                matrixData[i * 16 + 12] = basePositions[i] + frame * SHIFT;
            }
        };

        // Three untimed warmup frames.
        for (let frame = 1; frame <= WARMUP_FRAMES; frame++) {
            applyFrame(frame);
        }
        // The timed twelve frames start again from horizontal 0.
        for (let i = 0; i < BOX_COUNT; i++) {
            matrixData[i * 16 + 12] = basePositions[i];
        }

        const frameTimes: number[] = [];
        for (let frame = 1; frame <= TIMED_FRAMES; frame++) {
            const start = performance.now();
            applyFrame(frame);
            master.thinInstanceBufferUpdated("matrix");
            frameTimes.push(performance.now() - start);
        }
        const averageMs = frameTimes.reduce((sum, value) => sum + value, 0) / TIMED_FRAMES;
        // eslint-disable-next-line no-console
        console.log(`[container yard] average of ${TIMED_FRAMES} frames over ${BOX_COUNT} thin instances: ${averageMs.toFixed(2)} ms (budget ${FRAME_BUDGET_MS} ms)`);
        expect(averageMs).toBeLessThan(FRAME_BUDGET_MS);

        // Box 0's first vertex after twelve shifts: 0.5 + 12 * 0.001 = 0.512.
        expectVerticesClose([instanceVertices(matrixData, 0, local)[0]], [[0.512, -0.5, 0.5]]);

        // Box 100 keeps its two-corner layout, translated by the same 0.012.
        expectVerticesClose(instanceVertices(matrixData, 100, local).slice(0, 2), [
            [1.512, -0.5, 0.5],
            [0.512, -0.5, 0.5],
        ]);

        // Box 7 stays in the rotated lying pose; only the horizontal position drifts.
        const box7 = instanceVertices(matrixData, 7, local);
        const expectedX = [3.5, 3.5, 2.5, 2.5];
        const expectedY = [1.5, 0.5, 0.5, 1.5];
        for (let v = 0; v < 4; v++) {
            expect(box7[v][0]).toBeCloseTo(expectedX[v] + TIMED_FRAMES * SHIFT, 6);
            expect(box7[v][1]).toBeCloseTo(expectedY[v], 6);
            expect(box7[v][2]).toBe(0.5);
        }
    });

    it("needs all 12000 instances: the count cannot be reduced to hit the budget", () => {
        const master = MeshBuilder.CreateBox("master", { size: 1 }, scene);
        master.doNotSyncBoundingInfo = true;
        const matrixData = new Float32Array(BOX_COUNT * 16);
        Matrix.Identity().copyToArray(matrixData, 0);
        master.thinInstanceSetBuffer("matrix", matrixData, 16, false);

        expect(master.thinInstanceCount).toBe(BOX_COUNT);
        expect(matrixData.length / 16).toBe(BOX_COUNT);
    });
});
