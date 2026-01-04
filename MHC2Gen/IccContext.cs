using LittleCms;
using LittleCms.Data;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Buffers.Binary;
using System.IO;
using System.Text.Json;

namespace MHC2Gen
{
    class MHC2Tag
    {
        public const TagSignature Signature = (TagSignature)0x4D484332;

        public double MinCLL { get; set; }
        public double MaxCLL { get; set; }
        public double[,]? Matrix3x4 { get; set; }
        public double[,]? RegammaLUT { get; set; }


        public MHC2Tag() { }

        public MHC2Tag(ReadOnlySpan<byte> bytes)
        {
            LoadFromBytes(bytes);
        }

        private void LoadFromBytes(ReadOnlySpan<byte> bytes)
        {
            var ms0 = new MemoryStream(bytes.ToArray());
            var reader = new BinaryReader(ms0);

            // Signature
            reader.ReadBytes(4);
            // 0
            reader.ReadBytes(4);

            // Lut Size
            var lutSize = BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4));

            MinCLL = DecodeS15F16(BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4)));
            MaxCLL = DecodeS15F16(BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4)));

            // Matrix offset
            var maxtrixOffset = BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4));

            // Lut0 offset
            var lut0Offset = BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4));

            // Lut1 offset
            var lut1Offset = BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4));

            // Lut2 offset
            var lut2Offset = BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4));

            Matrix3x4 = new double[3, 4];

            for (var x = 0; x < 3; x++)
            {
                for (var y = 0; y < 4; y++)
                {
                    Matrix3x4[x, y] = DecodeS15F16(BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4)));
                }
            }

            RegammaLUT = new double[3, lutSize];

            for (var c = 0; c < 3; c++)
            {
                var discard = BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(8));

                for (var x = 0; x < lutSize; x++)
                {
                    RegammaLUT[c, x] = DecodeS15F16(BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4)));
                }
            }
        }

        public void ApplySdrAcm(double whiteLuminance = 120.0, double blackLuminance = 0.0, double gamma = 2.2, double boostPercentage = 0, double shadowDetailBoost = 0, double minCLL = 0.00248)
        {
            var lutSize = 4096; // Increased from 1024 to 4096 for maximum accuracy
            var amplifier = boostPercentage == 0 ? 1 : 1 + boostPercentage / 100;

            // Shadow dimming with three zones:
            // - Above linearThreshold: pure linear (no dimming)
            // - Between expOnlyThreshold and linearThreshold: blend from exponential to linear
            // - Below expOnlyThreshold: pure exponential dimming
            double shadowDimFactor = 1.0;             // 0.5 = half brightness at black
            shadowDimFactor += minCLL;
            const double linearThreshold = 1.0;       // Above this: no dimming
            const double expOnlyThreshold = 0.5;      // Below this: pure exponential
            const double expDecayRate = 3.0;          // Controls exponential steepness

            

            double blackLevelPQ = PQ(minCLL / 10000.0);

            RegammaLUT = new double[3, lutSize];

            for (var i = 0; i < lutSize; i++)
            {
                var (_, value) = CmsFunctions.SrgbAcm(i, whiteLuminance, blackLuminance, gamma, lutSize - 1);

                // Average between sSRGB and Piecewise gamma
                if (shadowDetailBoost > 0)
                {
                    var piecewiseValue = (double)i / (lutSize - 1);
                    var correctedBoost = (shadowDetailBoost * (lutSize - 1 - i)) / (lutSize - 1);
                    value = ((value * (100 - correctedBoost)) + (piecewiseValue * correctedBoost)) / 100;
                }

                value *= amplifier;

                // Convert to luminance
                double L_in = InversePQ(value) * 10000.0; // nits

                double finalValue;

                if (L_in <= minCLL)
                {
                    // Below black level - clamp
                    finalValue = blackLevelPQ;
                }
                else if (L_in >= linearThreshold)
                {
                    // Above linear threshold - no dimming, pass through
                    finalValue = value;
                }
                else
                {
                    // Calculate exponential-dimmed value
                    double distance = L_in - minCLL;
                    double dimMultiplier = 1.0 - (1.0 - shadowDimFactor) * Math.Exp(-expDecayRate * distance);
                    double L_exp = L_in * dimMultiplier;
                    double expValue = PQ(Math.Max(L_exp, minCLL) / 10000.0);

                    if (L_in <= expOnlyThreshold)
                    {
                        // Pure exponential zone
                        finalValue = expValue;
                    }
                    else
                    {
                        // Blend zone: smoothstep from exponential to linear
                        double blendT = (L_in - expOnlyThreshold) / (linearThreshold - expOnlyThreshold);
                        double smoothBlend = blendT * blendT * (3.0 - 2.0 * blendT);
                        finalValue = expValue * (1.0 - smoothBlend) + value * smoothBlend;
                    }
                }

                finalValue = Math.Clamp(finalValue, blackLevelPQ, 1.0);

                for (var c = 0; c < 3; c++)
                {
                    RegammaLUT[c, i] = finalValue;
                }
            }
        }

        public void ApplyPiecewise(double boostPercentage = 0, double minCLL = 0.00248)
        {
            var lutSize = 4096; // Increased from 1024 to 4096 for maximum accuracy
            var amplifier = boostPercentage == 0 ? 1 : 1 + boostPercentage / 100;

            // Calculate black level in PQ space
            double blackLevelPQ = PQ(minCLL / 10000.0);

            RegammaLUT = new double[3, lutSize];

            for (var i = 0; i < lutSize; i++)
            {
                var pqIn = (double)i / (lutSize - 1);

                // Convert to luminance, apply amplifier, convert back
                var luminance = InversePQ(pqIn) * 10000.0; // nits
                luminance *= amplifier;

                // Clamp to minCLL at the low end
                luminance = Math.Max(luminance, minCLL);

                var pqOut = PQ(luminance / 10000.0);
                pqOut = Math.Clamp(pqOut, blackLevelPQ, 1.0);

                for (var c = 0; c < 3; c++)
                {
                    RegammaLUT[c, i] = pqOut;
                }
            }
        }

        public void ApplyToneMappingCurve(double maxInputNits = 400, double maxOutputNits = 400, double curve_like = 400)
        {
            int lutSize = 4096; // Increased from 1024 to 4096 for maximum accuracy
            bool lutExist = true;
            if (RegammaLUT == null)
            {
                lutExist = false;
                RegammaLUT = new double[3, lutSize];
            }

            for (int i = 0; i < lutSize; i++)
            {
                double N = lutExist ? RegammaLUT[0, i] : (double)i / (lutSize - 1);
                double L = InversePQ(N) * 10000 * (maxInputNits / curve_like);
                double numerator = L * (maxInputNits + (L / Math.Pow(maxOutputNits / curve_like, 2)));
                double L_d = numerator / (maxInputNits + L);
                double N_prime = PQ(L_d / 10000);

                N_prime = Math.Max(0.0, Math.Min(1.0, N_prime));

                for (int c = 0; c < 3; c++)
                {
                    RegammaLUT[c, i] = N_prime;
                }
            }
        }

        public void ApplyToneMappingCurveGamma(double maxInputNits = 400, double maxOutputNits = 400, double curve_like = 400, double minCLL = 0.00248)
        {
            int lutSize = 4096;
            bool lutExist = true;
            if (RegammaLUT == null)
            {
                lutExist = false;
                RegammaLUT = new double[3, lutSize];
            }

            double L_target = (PQ(curve_like / 10000.0) / PQ(maxInputNits / 10000.0));
            double L_target_prime = (PQ(curve_like / 10000.0) / PQ(maxOutputNits / 10000.0));
            double difference = L_target_prime;

            double displayBlackLevelNits = minCLL;
            const double toeThresholdNits = 1.5;
            double blackLevelPQ = PQ(displayBlackLevelNits / 10000.0);
            double toeInputSignalPQ = PQ(toeThresholdNits / 10000.0);

            double toeInputSignalPQConverted = toeInputSignalPQ * difference;
            double toeLuminanceAfterGamma = InversePQ(toeInputSignalPQConverted) * 10000 * (maxInputNits / curve_like);
            double toeMappedPQ = PQ(toeLuminanceAfterGamma / 10000.0);

            double toeSpan = toeThresholdNits - displayBlackLevelNits;
            if (toeSpan <= 0) return; // safety

            for (int i = 0; i < lutSize; i++)
            {
                double N_in = lutExist ? RegammaLUT[0, i] : (double)i / (lutSize - 1); // existing or identity
                double L_in = InversePQ(N_in) * 10000.0;

                double N_prime;

                if (L_in <= displayBlackLevelNits)
                {
                    N_prime = blackLevelPQ;
                }
                else if (L_in <= toeThresholdNits)
                {
                    // Linear expansion from blackLevelPQ -> toeMappedPQ over 0.0007 .. 10 nits (no compression)
                    double t = (L_in - displayBlackLevelNits) / toeSpan; // 0..1
                    t = Math.Clamp(t, 0.0, 1.0);
                    N_prime = blackLevelPQ + t * (toeMappedPQ - blackLevelPQ);
                }
                else
                {
                    // Above toe: apply original gamma-style mapping (scaled domain)
                    double N_converted = N_in * difference;
                    double L_scaled = InversePQ(N_converted) * 10000 * (maxInputNits / curve_like);
                    N_prime = PQ(L_scaled / 10000.0);
                }

                // Clamp output
                N_prime = Math.Clamp(N_prime, 0.0, 1.0);

                for (int c = 0; c < 3; c++)
                {
                    RegammaLUT[c, i] = N_prime;
                }
            }
        }

        /// <summary>
        /// PQ-native tonemapping that replicates the gamma correction approach in pure PQ space.
        /// Works by applying a scale factor in PQ space, then converting to luminance and scaling again.
        /// This maintains saturation by working on each color channel independently.
        /// The contrastPivotNits parameter controls where the contrast boost is anchored.
        /// The brightnessMultiplier scales the output brightness (1.0 = no change, >1 = brighter).
        /// The maxBrightnessCompression compresses max brightness (1.0 = no compression, 0.25 = 25% of max).
        /// The minCLL sets the display black level in nits.
        /// </summary>
        public void ApplyToneMappingCurvePQNative(double maxInputNits = 400, double maxOutputNits = 400,
            double contrastPivotNits = 400, double brightnessMultiplier = 1.0, double maxBrightnessCompression = 1.0,
            double minCLL = 0.00248)
        {
            int lutSize = 4096;
            bool lutExist = true;
            if (RegammaLUT == null)
            {
                lutExist = false;
                RegammaLUT = new double[3, lutSize];
            }

            // Display characteristics
            double displayBlackLevelNits = minCLL;
            const double toeEndNits = 1.0; // Where toe curve ends
            const double toeBlendEndNits = 1.5; // Where blending to main curve completes
            const double toeGamma = 1.0; // <1 lifts shadows, >1 crushes them (1.0 = linear)

            // Shoulder region: start compressing at 70% of max output, fully compress by max input
            double shoulderStartNits = maxOutputNits * 1.0;

            // Calculate the PQ scaling factor
            // This replicates: difference = (PQ(gamma_like/10000) / PQ(maxOutputNits/10000))
            // Note: denominator uses maxOutputNits (not maxInputNits) - critical for correct slope!
            double scaleFactor = PQ(contrastPivotNits / 10000.0) / PQ(maxOutputNits / 10000.0);

            // Pre-calculate PQ values
            double blackLevelPQ = PQ(displayBlackLevelNits / 10000.0);
            double toeEndPQ = PQ(toeEndNits / 10000.0);

            // Calculate toe target using same transform as main curve
            double toeEndScaled = toeEndPQ * scaleFactor;
            double toeEndLuminance = InversePQ(toeEndScaled) * 10000.0 * (maxInputNits / contrastPivotNits);

            for (int i = 0; i < lutSize; i++)
            {
                double N_in = lutExist ? RegammaLUT[0, i] : (double)i / (lutSize - 1);
                double L_in = InversePQ(N_in) * 10000.0; // Convert to nits

                double N_out;

                if (L_in <= displayBlackLevelNits)
                {
                    // Black level clamp
                    N_out = blackLevelPQ;
                }
                else
                {
                    // Calculate toe curve value - interpolate in linear nits, then convert to PQ
                    // This respects the PQ perceptual curve shape
                    double toeT = Math.Clamp((L_in - displayBlackLevelNits) / (toeEndNits - displayBlackLevelNits), 0.0, 1.0);
                    double liftedT = Math.Pow(toeT, toeGamma);
                    double L_toe = displayBlackLevelNits + liftedT * (toeEndLuminance - displayBlackLevelNits);
                    // Apply brightness multiplier and compression to toe
                    L_toe *= brightnessMultiplier * maxBrightnessCompression;
                    L_toe = Math.Clamp(L_toe, displayBlackLevelNits, maxOutputNits * maxBrightnessCompression);
                    double N_toe = PQ(L_toe / 10000.0);

                    // Calculate main curve value
                    double N_scaled = N_in * scaleFactor;
                    double L_scaled = InversePQ(N_scaled) * 10000.0 * (maxInputNits / contrastPivotNits);

                    // Apply brightness multiplier and compression
                    L_scaled *= brightnessMultiplier * maxBrightnessCompression;

                    // Apply shoulder compression for smooth highlight rolloff
                    double compressedMaxOutput = maxOutputNits * maxBrightnessCompression;
                    double compressedShoulderStart = shoulderStartNits * maxBrightnessCompression;
                    if (L_scaled > compressedShoulderStart)
                    {
                        double t = (L_scaled - compressedShoulderStart) / (compressedMaxOutput - compressedShoulderStart);
                        t = Math.Clamp(t, 0.0, 1.0);
                        double smoothT = t / (t + (1.0 - t) * 0.5);
                        L_scaled = compressedShoulderStart + smoothT * (compressedMaxOutput - compressedShoulderStart);
                    }

                    L_scaled = Math.Clamp(L_scaled, displayBlackLevelNits, compressedMaxOutput);
                    double N_main = PQ(L_scaled / 10000.0);

                    // Blend between toe and main curve
                    if (L_in < toeEndNits)
                    {
                        // Pure toe region
                        N_out = N_toe;
                    }
                    else if (L_in < toeBlendEndNits)
                    {
                        // Blend zone: smoothly transition from toe to main curve
                        double blendT = (L_in - toeEndNits) / (toeBlendEndNits - toeEndNits);
                        double smoothBlend = blendT * blendT * (3.0 - 2.0 * blendT); // Smoothstep
                        N_out = N_toe * (1.0 - smoothBlend) + N_main * smoothBlend;
                    }
                    else
                    {
                        // Pure main curve
                        N_out = N_main;
                    }
                }

                // Clamp output to valid range
                N_out = Math.Clamp(N_out, 0.0, 1.0);

                for (int c = 0; c < 3; c++)
                {
                    RegammaLUT[c, i] = N_out;
                }
            }
        }

        /// <summary>
        /// Applies S-curve contrast in PQ space. Preserves black (0) and white (1) endpoints.
        /// contrastAmount: 1.0 = no change, >1 = more contrast, <1 = less contrast
        /// </summary>
        public void ApplyContrastSCurve(double contrastAmount = 1.0)
        {
            if (RegammaLUT == null || contrastAmount == 1.0)
                return;

            int lutSize = RegammaLUT.GetLength(1);

            for (int i = 0; i < lutSize; i++)
            {
                for (int c = 0; c < 3; c++)
                {
                    double pqValue = RegammaLUT[c, i];

                    // S-curve using power function anchored at 0.5
                    // For contrast > 1: steepens curve around midpoint
                    // For contrast < 1: flattens curve around midpoint
                    if (pqValue <= 0.0 || pqValue >= 1.0)
                        continue; // Preserve endpoints exactly

                    // Apply S-curve: shift to [-0.5, 0.5], apply contrast, shift back
                    double centered = pqValue - 0.5;
                    double sign = Math.Sign(centered);
                    double magnitude = Math.Abs(centered) * 2.0; // Scale to [0, 1]

                    // Apply power curve for contrast
                    double adjusted = Math.Pow(magnitude, 1.0 / contrastAmount);

                    // Scale back and shift
                    double result = 0.5 + sign * adjusted * 0.5;

                    RegammaLUT[c, i] = Math.Clamp(result, 0.0, 1.0);
                }
            }
        }

        public void ApplyWOLEDDesaturationCompensation(double compensationStrength)
        {
            if (RegammaLUT == null || compensationStrength <= 0)
                return;

            int lutSize = RegammaLUT.GetLength(1);

            var originalLUT = new double[3, lutSize];
            for (int c = 0; c < 3; c++)
            {
                for (int i = 0; i < lutSize; i++)
                {
                    originalLUT[c, i] = RegammaLUT[c, i];
                }
            }

            // Apply HDR-aware WOLED desaturation compensation
            // This works in the PQ (ST2084) domain for HDR content in Rec2020
            for (int i = 0; i < lutSize; i++)
            {
                double currentR = originalLUT[0, i];
                double currentG = originalLUT[1, i];
                double currentB = originalLUT[2, i];

                double linearR = InversePQ(currentR) * 10000;
                double linearG = InversePQ(currentG) * 10000;
                double linearB = InversePQ(currentB) * 10000;

                // Rec.2020 luminance coefficients
                double luminance = 0.2627 * linearR + 0.6780 * linearG + 0.0593 * linearB;

                double luminanceNormalized = Math.Min(luminance / 10000.0, 1.0);
                double luminanceFactor = Math.Pow(1.0 - luminanceNormalized, 1.5);

                double adaptiveCompensation = (compensationStrength / 100.0) * luminanceFactor * 0.3;

                if (luminance > 0.01)
                {
                    double rRatio = linearR / luminance;
                    double gRatio = linearG / luminance;
                    double bRatio = linearB / luminance;

                    double saturationFactor = 1.0 + adaptiveCompensation;

                    double enhancedR = luminance * (rRatio * saturationFactor + (1.0 - saturationFactor) / 3.0);
                    double enhancedG = luminance * (gRatio * saturationFactor + (1.0 - saturationFactor) / 3.0);
                    double enhancedB = luminance * (bRatio * saturationFactor + (1.0 - saturationFactor) / 3.0);

                    enhancedR = Math.Max(0.0, Math.Min(enhancedR, 10000.0));
                    enhancedG = Math.Max(0.0, Math.Min(enhancedG, 10000.0));
                    enhancedB = Math.Max(0.0, Math.Min(enhancedB, 10000.0));

                    RegammaLUT[0, i] = PQ(enhancedR / 10000.0);
                    RegammaLUT[1, i] = PQ(enhancedG / 10000.0);
                    RegammaLUT[2, i] = PQ(enhancedB / 10000.0);
                }
                else
                {
                    RegammaLUT[0, i] = currentR;
                    RegammaLUT[1, i] = currentG;
                    RegammaLUT[2, i] = currentB;
                }

                RegammaLUT[0, i] = Math.Max(0.0, Math.Min(1.0, RegammaLUT[0, i]));
                RegammaLUT[1, i] = Math.Max(0.0, Math.Min(1.0, RegammaLUT[1, i]));
                RegammaLUT[2, i] = Math.Max(0.0, Math.Min(1.0, RegammaLUT[2, i]));
            }
        }

        // PQ EOTF function: converts luminance (cd/m^2) to normalized signal value
        private double PQ(double L)
        {
            double m1 = 0.1593017578125;
            double m2 = 78.84375;
            double c1 = 0.8359375;
            double c2 = 18.8515625;
            double c3 = 18.6875;

            double Lm1 = Math.Pow(L, m1);
            double numerator = c1 + c2 * Lm1;
            double denominator = 1 + c3 * Lm1;
            double N = Math.Pow(numerator / denominator, m2);

            return N;
        }

        // Inverse PQ EOTF function: converts normalized signal value to luminance (cd/m^2)
        private double InversePQ(double N)
        {
            double m1 = 0.1593017578125;
            double m2 = 78.84375;
            double c1 = 0.8359375;
            double c2 = 18.8515625;
            double c3 = 18.6875;

            double N1_m2 = Math.Pow(N, 1.0 / m2);
            double numerator = N1_m2 - c1;
            double denominator = c2 - c3 * N1_m2;

            double Lm1 = numerator / denominator;

            // Ensure Lm1 is non-negative to avoid invalid values
            Lm1 = Math.Max(Lm1, 0.0);

            double L = Math.Pow(Lm1, 1.0 / m1);

            return L;
        }



        public void ApplyGamma(double gamma = 2.4, double shadowDetailBoost = 0)
        {
            var lutSize = 1024;

            RegammaLUT = new double[3, lutSize];

            for (var i = 0; i < lutSize; i++)
            {
                var value = CmsFunctions.RgbToLinear((double)i / (lutSize - 1), gamma);

                // Average between sSRGB and Piecewise gamma
                if (shadowDetailBoost > 0)
                {
                    var piecewiseValue = (double)i / lutSize;

                    var correctedBoost = (shadowDetailBoost * (lutSize - i)) / lutSize;

                    value = ((value * (100 - correctedBoost)) + (piecewiseValue * correctedBoost)) / 100;
                }

                for (var c = 0; c < 3; c++)
                {
                    RegammaLUT[c, i] = value;
                }
            }
        }

        private static int EncodeS15F16(double value)
        {
            var x = (int)Math.Round(value * 65536);
            return x;
        }

        private static double DecodeS15F16(int value)
        {
            return (double)value / 65536;
        }

        public byte[] ToBytes()
        {
            if (RegammaLUT!.GetLength(0) != 3) throw new ArrayTypeMismatchException();
            var lut_size = RegammaLUT.GetLength(1);
            if (lut_size <= 1 || lut_size > 4096) throw new IndexOutOfRangeException();
            if (Matrix3x4!.Length != 12 || Matrix3x4.GetLength(0) != 3) throw new ArrayTypeMismatchException();
            var ms0 = new MemoryStream();
            var writer = new BinaryWriter(ms0);
            // type
            writer.Write(BinaryPrimitives.ReverseEndianness((int)Signature));
            writer.Write(BinaryPrimitives.ReverseEndianness(0));

            writer.Write(BinaryPrimitives.ReverseEndianness(lut_size));
            writer.Write(BinaryPrimitives.ReverseEndianness(EncodeS15F16(MinCLL)));
            writer.Write(BinaryPrimitives.ReverseEndianness(EncodeS15F16(MaxCLL)));
            // matrix offset
            writer.Write(BinaryPrimitives.ReverseEndianness(36));
            // lut0 offset
            writer.Write(BinaryPrimitives.ReverseEndianness(84));
            // lut1 offset
            var lut1_offset = 84 + 8 + lut_size * 4;
            writer.Write(BinaryPrimitives.ReverseEndianness(lut1_offset));
            // lut2 offset
            var lut2_offset = lut1_offset + 8 + lut_size * 4;
            writer.Write(BinaryPrimitives.ReverseEndianness(lut2_offset));

            foreach (var e in Matrix3x4)
            {
                writer.Write(BinaryPrimitives.ReverseEndianness(EncodeS15F16(e)));
            }

            for (int ch = 0; ch < 3; ch++)
            {
                writer.Write(new ReadOnlySpan<byte>(new byte[] { (byte)'s', (byte)'f', (byte)'3', (byte)'2', 0, 0, 0, 0 }));
                for (int i = 0; i < lut_size; i++)
                {
                    writer.Write(BinaryPrimitives.ReverseEndianness(EncodeS15F16(RegammaLUT[ch, i])));
                }
            }
            writer.Flush();
            return ms0.ToArray();
        }

        internal static MHC2Tag? LoadFromProfile(IccProfile profile)
        {
            var bytes = profile.RawBytes;

            if (bytes == null)
            {
                return null;
            }

            // 0x4D484332
            var index = -1;
            var first = true;
            for (var i = 0; i < bytes.Length; i += 4)
            {
                var chunk = bytes.AsSpan(new Range(i, i + 4));

                if (chunk[0] == 77 && chunk[1] == 72 && chunk[2] == 67 && chunk[3] == 50)
                {
                    index = i;

                    if (first)
                    {
                        index = -1;
                        first = false;
                        continue;
                    }
                    break;
                }
            }

            if (index == -1)
            {
                return null;
            }

            var mhc2Bytes = bytes.AsSpan(Range.StartAt(index));

            return new MHC2Tag(mhc2Bytes);
        }
    }

    internal class ExtraInfoTag
    {
        public SDRTransferFunction SDRTransferFunction { get; set; }
        public double Gamma { get; set; }
        public double SDRMinBrightness { get; set; }
        public double SDRMaxBrightness { get; set; }
        public double SDRBrightnessBoost { get; set; }
        public double ShadowDetailBoost { get; set; }
        public ColorGamut TargetGamut { get; set; }
        public double ToneMappingFromLuminance { get; set; }
        public double ToneMappingToLuminance { get; set; }

        public double HdrGammaMultiplier { get; set; } = 1.0;
        public double HdrBrightnessMultiplier { get; set; } = 1.0;
        public double WOLEDDesaturationCompensation { get; set; }
    }

    internal class IccContext
    {
        protected IccProfile profile;
        public CIEXYZ IlluminantRelativeWhitePoint { get; }
        public Matrix<double>? ChromaticAdaptionMatrix { get; }
        public Matrix<double>? InverseChromaticAdaptionMatrix { get; }
        public RgbPrimaries ProfilePrimaries { get; }
        public MHC2Tag? MHC2 { get; set; }

        public IccContext(IccProfile profile)
        {
            if (profile.PCS != ColorSpaceSignature.XYZ || profile.ColorSpace != ColorSpaceSignature.Rgb)
            {
                throw new CmsException(CmsError.COLORSPACE_CHECK, "ICC profile is not XYZ->RGB");
            }
            this.profile = profile;
            {
                if (profile.TryReadTag(SafeTagSignature.ChromaticAdaptationTag, out var chad))
                    ChromaticAdaptionMatrix = DenseMatrix.OfArray(chad);
                InverseChromaticAdaptionMatrix = ChromaticAdaptionMatrix?.Inverse();
            }
            (IlluminantRelativeWhitePoint, ProfilePrimaries) = PopulatePrimaries();

            MHC2 = MHC2Tag.LoadFromProfile(profile);
        }

        private unsafe CIEXYZ GetIlluminantReletiveWhitePoint()
        {
            if (profile.TryReadTag(SafeTagSignature.MediaWhitePointTag, out var icc_wtpt))
            {
                if (ChromaticAdaptionMatrix == null || profile.HeaderCreator == 0x6170706c /* 'aapl' */)
                {
                    // for profiels without 'chad' tag and Apple profiles, mediaWhitepointTag is illuminant-relative
                    return icc_wtpt;
                }
                else
                {
                    // ... otherwise it is PCS-relative
                    var pcs_wtpt = icc_wtpt;
                    if (ChromaticAdaptionMatrix != null)
                    {
                        return ApplyInverseChad(pcs_wtpt);
                    }
                }
            }
            if (ChromaticAdaptionMatrix != null)
            {
                // no wtpt in icc, sum RGB and reverse chad
                var pcs_rXYZ = profile.ReadTag(SafeTagSignature.RedColorantTag);
                var pcs_gXYZ = profile.ReadTag(SafeTagSignature.GreenColorantTag);
                var pcs_bXYZ = profile.ReadTag(SafeTagSignature.BlueColorantTag);
                var pcs_sumrgb = pcs_rXYZ + pcs_gXYZ + pcs_bXYZ;

                return ApplyInverseChad(pcs_sumrgb);
            }
            else
            {
                throw new Exception("malformed profile: missing wtpt and chad");
            }
        }

        protected CIEXYZ ApplyInverseChad(in CIEXYZ val)
        {
            var vec = InverseChromaticAdaptionMatrix!.Multiply(new DenseVector(new[] { val.X, val.Y, val.Z }));
            return new() { X = vec[0], Y = vec[1], Z = vec[2] };
        }


        /// <summary>
        /// use lcms transform to get illuminant-relative primaries.
        /// </summary>
        private unsafe (CIEXYZ, RgbPrimaries) PopulatePrimaries()
        {
            var ctx = new CmsContext();

            ctx.SetAdaptionState(0);

            var xyzprof = IccProfile.CreateXYZ();
            var t = new CmsTransform(ctx, profile, CmsPixelFormat.RGBDouble, xyzprof, CmsPixelFormat.XYZDouble, RenderingIntent.ABSOLUTE_COLORIMETRIC, default);
            var pixels = new ReadOnlySpan<double>(new double[] {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                1, 1, 1,
            });
            Span<double> xyz = stackalloc double[3];


            t.DoTransform(pixels.Slice(0), xyz, 1);
            var rXYZ = new CIEXYZ { X = xyz[0], Y = xyz[1], Z = xyz[2] };
            t.DoTransform(pixels.Slice(3), xyz, 1);
            var gXYZ = new CIEXYZ { X = xyz[0], Y = xyz[1], Z = xyz[2] };
            t.DoTransform(pixels.Slice(6), xyz, 1);
            var bXYZ = new CIEXYZ { X = xyz[0], Y = xyz[1], Z = xyz[2] };
            t.DoTransform(pixels.Slice(9), xyz, 1);
            var wXYZ = new CIEXYZ { X = xyz[0], Y = xyz[1], Z = xyz[2] };


            return (wXYZ, new(rXYZ.ToXY(), gXYZ.ToXY(), bXYZ.ToXY(), wXYZ.ToXY()));

        }

        public string GetDescription()
        {
            return profile.GetInfo(InfoType.Description);
        }

        public CIEXYZ GetIlluminantRelativeBlackPoint()
        {
            // NOTE: mediaBlackPointTag is no longer in ICC standard
            if (profile.TryReadTag(SafeTagSignature.MediaBlackPointTag, out var bkpt))
            {
                // no chad in profile, bkpt is illuminant-relative
                if (ChromaticAdaptionMatrix == null)
                {
                    return bkpt;
                }
                else
                {
                    return ApplyInverseChad(bkpt);
                }
            }

            // no bkpt in tag, use lcms transform
            var ctx = new CmsContext();
            ctx.SetAdaptionState(0);
            var t = new CmsTransform(ctx, profile, CmsPixelFormat.RGB8, IccProfile.CreateXYZ(), CmsPixelFormat.XYZDouble, RenderingIntent.ABSOLUTE_COLORIMETRIC, default);
            var input = new ReadOnlySpan<byte>(new byte[] { 0, 0, 0 });
            Span<double> outbuf = stackalloc double[3];
            t.DoTransform(input, outbuf, 1);
            bkpt = new CIEXYZ { X = outbuf[0], Y = outbuf[1], Z = outbuf[2] };
            return bkpt;
        }

        public void WriteIlluminantRelativeMediaBlackPoint(in CIEXYZ value)
        {
            CIEXYZ valueToWrite;
            if (ChromaticAdaptionMatrix != null)
            {
                var vec = new DenseVector(new double[] { value.X, value.Y, value.Z });
                var pcs_vec = ChromaticAdaptionMatrix * vec;
                valueToWrite = new() { X = pcs_vec[0], Y = pcs_vec[1], Z = pcs_vec[2] };
            }
            else
            {
                valueToWrite = value;
            }
            profile.WriteTag(SafeTagSignature.MediaBlackPointTag, valueToWrite);
        }

        public static Matrix<double> GetChromaticAdaptationMatrix(CIEXYZ sourceIlluminant, CIEXYZ targetIlluminant)
        {
            // http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
            // Bradford
            var M_a = DenseMatrix.OfArray(new[,] {
                { 0.8951, 0.2664, -0.1614 },
                { -0.7502, 1.7135, 0.0367 },
                { 0.0389, -0.0685, 1.0296 },
            });

            var M_a_inv = M_a.Inverse();

            var cone_s_vec = M_a * new DenseVector(new double[] { sourceIlluminant.X, sourceIlluminant.Y, sourceIlluminant.Z });
            var cone_t_vec = M_a * new DenseVector(new double[] { targetIlluminant.X, targetIlluminant.Y, targetIlluminant.Z });

            var M = M_a_inv * DenseMatrix.Build.DenseOfDiagonalVector(cone_t_vec / cone_s_vec) * M_a;

            return M;
        }
    }

    internal class DeviceIccContext : IccContext
    {
        CIEXYZ illuminantRelativeBlackPoint;
        public double min_nits;
        public double max_nits;
        ToneCurve profileRedToneCurve;
        ToneCurve profileGreenToneCurve;
        ToneCurve profileBlueToneCurve;
        ToneCurve profileRedReverseToneCurve;
        ToneCurve profileGreenReverseToneCurve;
        ToneCurve profileBlueReverseToneCurve;

        public ExtraInfoTag? ExtraInfoTag { get; }

        public bool UseChromaticAdaptation { get; set; }

        public DeviceIccContext(IccProfile profile) : base(profile)
        {
            illuminantRelativeBlackPoint = GetIlluminantRelativeBlackPoint();
            (max_nits, min_nits) = GetProfileLuminance();
            profileRedToneCurve = profile.ReadTag(SafeTagSignature.RedTRCTag);
            profileGreenToneCurve = profile.ReadTag(SafeTagSignature.GreenTRCTag);
            profileBlueToneCurve = profile.ReadTag(SafeTagSignature.BlueTRCTag);
            profileRedReverseToneCurve = profileRedToneCurve.Reverse();
            profileGreenReverseToneCurve = profileGreenToneCurve.Reverse();
            profileBlueReverseToneCurve = profileBlueToneCurve.Reverse();

            if (profile.ContainsTag(TagSignature.ScreeningDescTag))
            {
                var ccDesc = profile.ReadTag(SafeTagSignature.ScreeningDescTag);
                var json = ccDesc?.Get("en", "US");
                if (json != null)
                {
                    ExtraInfoTag = JsonSerializer.Deserialize<ExtraInfoTag>(json);
                }
            }
        }

        public static Matrix<double> RgbToXYZ(RgbPrimaries primaries)
        {
            var rXYZ = primaries.Red.ToXYZ();
            var gXYZ = primaries.Green.ToXYZ();
            var bXYZ = primaries.Blue.ToXYZ();
            var wXYZ = primaries.White.ToXYZ();

            var S = DenseMatrix.OfArray(new[,] {
                {rXYZ.X, gXYZ.X, bXYZ.X},
                {rXYZ.Y, gXYZ.Y, bXYZ.Y},
                {rXYZ.Z, gXYZ.Z, bXYZ.Z},
            }).Inverse().Multiply(DenseMatrix.OfArray(new[,] { { wXYZ.X }, { wXYZ.Y }, { wXYZ.Z } }));

            var M = DenseMatrix.OfArray(new[,] {
                {S[0,0] * rXYZ.X, S[1,0]*gXYZ.X, S[2,0]*bXYZ.X },
                {S[0,0] * rXYZ.Y, S[1,0]*gXYZ.Y, S[2,0]*bXYZ.Y },
                {S[0,0] * rXYZ.Z, S[1,0]*gXYZ.Z, S[2,0]*bXYZ.Z },
            });
            return M;
        }

        public static Matrix<double> XYZToRgb(RgbPrimaries primaries) => RgbToXYZ(primaries).Inverse();

        public static Matrix<double> RgbToRgb(RgbPrimaries from, RgbPrimaries to)
        {
            var M1 = RgbToXYZ(from);
            var M2 = XYZToRgb(to);
            return M2 * M1;
        }

        /// <summary>
        /// Creates Rec2020 primaries adapted to the device's native color characteristics.
        /// Calculates the delta between native primaries and sRGB/Rec.709, then SUBTRACTS to compensate.
        /// This corrects for the display's deviation while providing Rec2020's expanded gamut.
        /// Uses sRGB as base since most SDR content (YouTube, etc.) is mastered for sRGB/Rec.709.
        /// </summary>
        private static RgbPrimaries CalculateRec2020Native(RgbPrimaries devicePrimaries)
        {
            var srgb = RgbPrimaries.sRGB;
            var rec2020 = RgbPrimaries.Rec2020;

            // Compensate: Rec2020 - (Native - sRGB) = Rec2020 + (sRGB - Native)
            // If display renders red more saturated than sRGB, pull Rec2020 red back
            return new RgbPrimaries(
                new CIExy { x = rec2020.Red.x + (srgb.Red.x - devicePrimaries.Red.x),
                            y = rec2020.Red.y + (srgb.Red.y - devicePrimaries.Red.y) },
                new CIExy { x = rec2020.Green.x + (srgb.Green.x - devicePrimaries.Green.x),
                            y = rec2020.Green.y + (srgb.Green.y - devicePrimaries.Green.y) },
                new CIExy { x = rec2020.Blue.x + (srgb.Blue.x - devicePrimaries.Blue.x),
                            y = rec2020.Blue.y + (srgb.Blue.y - devicePrimaries.Blue.y) },
                rec2020.White  // Keep D65 white point protected
            );
        }

        public string GetDeviceDescription()
        {
            var model = profile.GetInfo(InfoType.Model);
            if (!string.IsNullOrEmpty(model)) return model;
            var desc = GetDescription();
            if (!string.IsNullOrEmpty(desc)) return desc;
            return "<Unknown device>";
        }

        private (double MaxNits, double MinNits) GetProfileLuminance()
        {
            var wtpt = IlluminantRelativeWhitePoint;
            double max_nits = 80;
            if (profile.TryReadTag(SafeTagSignature.LuminanceTag, out var lumi))
            {
                max_nits = lumi.Y;
            }
            var min_nits = 0.0;
            var bkpt = illuminantRelativeBlackPoint;
            if (bkpt.Y != 0)
            {
                var bkpt_scale = bkpt.Y / wtpt.Y;
                min_nits = max_nits * bkpt_scale;
            }
            return (max_nits, min_nits);
        }

        public IccProfile CreateMhc2CscIcc(RgbPrimaries? sourcePrimaries = null, string sourceDescription = "sRGB")
        {
            var wtpt = IlluminantRelativeWhitePoint;
            var vcgt = profile.ReadTagOrDefault(SafeTagSignature.VcgtTag)?.ToArray();

            var devicePrimaries = ProfilePrimaries;

            var deviceOetf = new ToneCurve[] { profileRedReverseToneCurve, profileGreenReverseToneCurve, profileBlueReverseToneCurve };

            var srgbTrc = IccProfile.Create_sRGB().ReadTag(SafeTagSignature.RedTRCTag)!;
            var sourceEotf = new ToneCurve[] { srgbTrc, srgbTrc, srgbTrc };

            sourcePrimaries ??= RgbPrimaries.sRGB;

            var srgb_to_xyz = RgbToXYZ(RgbPrimaries.sRGB);
            var xyz_to_srgb = XYZToRgb(RgbPrimaries.sRGB);


            Matrix<double> user_matrix = DenseMatrix.CreateIdentity(3);

            // pipeline here: input signal converted to XYZ (interpreted as sRGB)

            if (!ReferenceEquals(sourcePrimaries, RgbPrimaries.sRGB))
            {
                user_matrix = RgbToXYZ(sourcePrimaries) * xyz_to_srgb * user_matrix;
            }

            // pipeline here: input signal converted to XYZ (interpreted as custom RGB)

            if (UseChromaticAdaptation)
            {
                user_matrix = GetChromaticAdaptationMatrix(sourcePrimaries.White.ToXYZ(), devicePrimaries.White.ToXYZ()) * user_matrix;
            }

            // pipeline here: input signal XYZ adapted to device white point

            // hook: scale white point

            var source_white_to_xyz = user_matrix * new DenseVector(new double[] { 1, 1, 1 });
            var mapped_y = source_white_to_xyz[1];
            var profile_max_nits = max_nits * (mapped_y / wtpt.Y);

            // end hook

            user_matrix = XYZToRgb(devicePrimaries) * user_matrix;

            // pipeline here: linear device RGB

            // hack: eliminate fixed sRGB to XYZ transform

            var mhc2_matrix = new double[,] {
               { user_matrix[0,0], user_matrix[0,1], user_matrix[0,2], 0 },
               { user_matrix[1,0], user_matrix[1,1], user_matrix[1,2], 0 },
               { user_matrix[2,0], user_matrix[2,1], user_matrix[2,2], 0 },
            };

            double[,] mhc2_lut;
            if (vcgt != null)
            {
                var lut_size = 1024;
                mhc2_lut = new double[3, lut_size];
                for (int ch = 0; ch < 3; ch++)
                {
                    for (int iinput = 0; iinput < lut_size; iinput++)
                    {
                        var input = (float)iinput / (lut_size - 1);
                        var linear = sourceEotf[ch].EvalF32(input);
                        var dev_output = deviceOetf[ch].EvalF32(linear);
                        if (vcgt != null)
                        {
                            dev_output = vcgt[ch].EvalF32(dev_output);
                        }
                        mhc2_lut[ch, iinput] = dev_output;
                    }
                }
            }
            else
            {
                mhc2_lut = new double[,] 
                { 
                    { 0, 1 }, 
                    { 0, 1 }, 
                    { 0, 1 }, 
                };
            }

            var mhc2d = new MHC2Tag
            {
                MinCLL = min_nits,
                MaxCLL = max_nits,
                Matrix3x4 = mhc2_matrix,
                RegammaLUT = mhc2_lut
            };

            var mhc2 = mhc2d.ToBytes();

            var outputProfile = IccProfile.CreateRGB(sourcePrimaries.White.ToXYZ().ToCIExyY(), new CIExyYTRIPLE
            {
                Red = sourcePrimaries.Red.ToXYZ().ToCIExyY(),
                Green = sourcePrimaries.Green.ToXYZ().ToCIExyY(),
                Blue = sourcePrimaries.Blue.ToXYZ().ToCIExyY()
            }, new RgbToneCurve(srgbTrc, srgbTrc, srgbTrc));

            outputProfile.WriteTag(SafeTagSignature.LuminanceTag, new CIEXYZ { Y = profile_max_nits });

            var outctx = new IccContext(outputProfile);
            outctx.WriteIlluminantRelativeMediaBlackPoint(illuminantRelativeBlackPoint);

            // copy device description from device profile
            var copy_tags = new TagSignature[] { TagSignature.DeviceMfgDescTag, TagSignature.DeviceModelDescTag };

            unsafe
            {
                foreach (var tag in copy_tags)
                {
                    var tag_ptr = profile.ReadTag(tag);
                    if (tag_ptr != null)
                    {
                        outputProfile.WriteTag(tag, tag_ptr);
                    }
                }
            }

            // set output profile description
            outputProfile.HeaderManufacturer = profile.HeaderManufacturer;
            outputProfile.HeaderModel = profile.HeaderModel;
            outputProfile.HeaderAttributes = profile.HeaderAttributes;
            outputProfile.HeaderRenderingIntent = RenderingIntent.ABSOLUTE_COLORIMETRIC;

            var new_desc = $"CSC: {sourceDescription} ({GetDeviceDescription()})";
            var new_desc_mlu = new MLU(new_desc);
            outputProfile.WriteTag(SafeTagSignature.ProfileDescriptionTag, new_desc_mlu);

            outputProfile.WriteRawTag(MHC2Tag.Signature, mhc2);

            outputProfile.ComputeProfileId();

            return outputProfile;
        }

        public IccProfile CreatePQ10DecodeIcc(double? maxBrightnessOverride = null, double? minBrightnessOverride = null)
        {
            var sourcePrimaries = RgbPrimaries.Rec2020;
            var devicePrimaries = ProfilePrimaries;

            Matrix<double> user_matrix = DenseMatrix.CreateIdentity(3);


            if (UseChromaticAdaptation)
            {
                user_matrix = GetChromaticAdaptationMatrix(sourcePrimaries.White.ToXYZ(), devicePrimaries.White.ToXYZ()) * user_matrix;
            }


            // var rgb_transform = RgbToRgb(sourcePrimaries, devicePrimaries);
            // rgb_transform = XYZToRgb(devicePrimaries) * RgbToXYZ(sourcePrimaries);
            // var xyz_transform = RgbToXYZ(sourcePrimaries) * rgb_transform * XYZToRgb(sourcePrimaries);
            user_matrix = RgbToXYZ(sourcePrimaries) * XYZToRgb(devicePrimaries) * user_matrix;

            var mhc2_matrix = new double[,] {
               { user_matrix[0,0], user_matrix[0,1], user_matrix[0,2], 0 },
               { user_matrix[1,0], user_matrix[1,1], user_matrix[1,2], 0 },
               { user_matrix[2,0], user_matrix[2,1], user_matrix[2,2], 0 },
            };

            var vcgt = profile.ReadTagOrDefault(SafeTagSignature.VcgtTag)?.ToArray();
            var deviceOetf = new ToneCurve[] { profileRedReverseToneCurve, profileGreenReverseToneCurve, profileBlueReverseToneCurve };

            var use_max_nits = maxBrightnessOverride ?? max_nits;
            var use_min_nits = minBrightnessOverride ?? min_nits;

            var lut_size = 4096;
            var mhc2_lut = new double[3, 4096];
            for (int ch = 0; ch < 3; ch++)
            {
                for (int iinput = 0; iinput < lut_size; iinput++)
                {
                    var pqinput = (double)iinput / (lut_size - 1);
                    var nits = ST2084.SignalToNits(pqinput);
                    var linear = Math.Max(nits - use_min_nits, 0) / (use_max_nits - use_min_nits);
                    var dev_output = deviceOetf[ch].EvalF32((float)linear);
                    if (vcgt != null)
                    {
                        dev_output = vcgt[ch].EvalF32(dev_output);
                    }
                    // Console.WriteLine($"Channel {ch}: PQ {iinput} -> {nits} cd/m2 -> SDR {dev_output * 255}");
                    mhc2_lut[ch, iinput] = dev_output;
                }
            }

            var mhc2d = new MHC2Tag
            {
                MinCLL = use_min_nits,
                MaxCLL = use_max_nits,
                Matrix3x4 = mhc2_matrix,
                RegammaLUT = mhc2_lut
            };

            var mhc2 = mhc2d.ToBytes();

            var outputProfile = IccProfile.CreateRGB(devicePrimaries.White.ToXYZ().ToCIExyY(), new CIExyYTRIPLE
            {
                Red = devicePrimaries.Red.ToXYZ().ToCIExyY(),
                Green = devicePrimaries.Green.ToXYZ().ToCIExyY(),
                Blue = devicePrimaries.Blue.ToXYZ().ToCIExyY()
            }, new RgbToneCurve(profileRedToneCurve, profileGreenToneCurve, profileBlueToneCurve));

            // copy characteristics from device profile
            var copy_tags = new TagSignature[] { TagSignature.DeviceMfgDescTag, TagSignature.DeviceModelDescTag };
            unsafe
            {
                foreach (var tag in copy_tags)
                {
                    var tag_ptr = profile.ReadTag(tag);
                    if (tag_ptr != null)
                    {
                        outputProfile.WriteTag(tag, tag_ptr);
                    }
                }
            }

            outputProfile.WriteTag(SafeTagSignature.LuminanceTag, new CIEXYZ { Y = use_max_nits });

            // the profile is not read by regular applications
            // var outctx = new IccContext(outputProfile);
            // outctx.WriteIlluminantRelativeMediaBlackPoint(illuminantRelativeBlackPoint);

            // set output profile description
            outputProfile.HeaderManufacturer = profile.HeaderManufacturer;
            outputProfile.HeaderModel = profile.HeaderModel;
            outputProfile.HeaderAttributes = profile.HeaderAttributes;
            outputProfile.HeaderRenderingIntent = RenderingIntent.ABSOLUTE_COLORIMETRIC;

            var new_desc = $"HDR10 to SDR ({GetDeviceDescription()}, {use_max_nits:0} nits)";
            var new_desc_mlu = new MLU(new_desc);
            outputProfile.WriteTag(SafeTagSignature.ProfileDescriptionTag, new_desc_mlu);

            outputProfile.WriteRawTag(MHC2Tag.Signature, mhc2);

            outputProfile.ComputeProfileId();

            return outputProfile;
        }

        public IccProfile CreateSdrAcmIcc(bool calibrateTransfer)
        {
            Matrix<double> user_matrix = DenseMatrix.CreateIdentity(3);

            if (UseChromaticAdaptation)
            {
                user_matrix = GetChromaticAdaptationMatrix(new CIExy { x = 0.3127, y = 0.3290 }.ToXYZ(), ProfilePrimaries.White.ToXYZ()) * user_matrix;
            }

            var mhc2_matrix = new double[,] {
               { user_matrix[0,0], user_matrix[0,1], user_matrix[0,2], 0 },
               { user_matrix[1,0], user_matrix[1,1], user_matrix[1,2], 0 },
               { user_matrix[2,0], user_matrix[2,1], user_matrix[2,2], 0 },
            };

            double[,] mhc2_lut;

            var outputprofileTrc = new RgbToneCurve(profileRedToneCurve, profileGreenToneCurve, profileBlueToneCurve);
            var vcgt = profile.ReadTagOrDefault(SafeTagSignature.VcgtTag)?.ToArray();

            if (calibrateTransfer)
            {
                var sourceEotf = IccProfile.Create_sRGB().ReadTag(SafeTagSignature.RedTRCTag);
                outputprofileTrc = new RgbToneCurve(sourceEotf, sourceEotf, sourceEotf);

                var deviceOetf = new ToneCurve[] { profileRedReverseToneCurve, profileGreenReverseToneCurve, profileBlueReverseToneCurve };
                var lut_size = 1024;
                mhc2_lut = new double[3, lut_size];
                for (int ch = 0; ch < 3; ch++)
                {
                    for (int iinput = 0; iinput < lut_size; iinput++)
                    {
                        var input = (float)iinput / (lut_size - 1);
                        var linear = sourceEotf.EvalF32(input);
                        var dev_output = deviceOetf[ch].EvalF32(linear);
                        if (vcgt != null)
                        {
                            dev_output = vcgt[ch].EvalF32(dev_output);
                        }
                        mhc2_lut[ch, iinput] = dev_output;
                    }

                }
            }
            else if (vcgt != null)
            {
                // move vcgt to mhc2 only
                var lut_size = 1024;
                mhc2_lut = new double[3, lut_size];
                for (int ch = 0; ch < 3; ch++)
                {
                    for (int iinput = 0; iinput < lut_size; iinput++)
                    {
                        var input = (float)iinput / (lut_size - 1);
                        var dev_output = vcgt[ch].EvalF32(input);
                        mhc2_lut[ch, iinput] = dev_output;
                    }
                }
            }
            else
            {
                mhc2_lut = new double[,] { { 0, 1 }, { 0, 1 }, { 0, 1 } };
            }


            var mhc2d = new MHC2Tag
            {
                MinCLL = min_nits,
                MaxCLL = max_nits,
                Matrix3x4 = mhc2_matrix,
                RegammaLUT = mhc2_lut
            };

            var mhc2 = mhc2d.ToBytes();

            var devicePrimaries = ProfilePrimaries;
            var outputProfile = IccProfile.CreateRGB(devicePrimaries.White.ToXYZ().ToCIExyY(), new CIExyYTRIPLE
            {
                Red = devicePrimaries.Red.ToXYZ().ToCIExyY(),
                Green = devicePrimaries.Green.ToXYZ().ToCIExyY(),
                Blue = devicePrimaries.Blue.ToXYZ().ToCIExyY()
            }, new RgbToneCurve(profileRedToneCurve, profileGreenToneCurve, profileBlueToneCurve));

            outputProfile.WriteTag(SafeTagSignature.LuminanceTag, new CIEXYZ { Y = max_nits });

            // the profile is not read by regular applications
            // var outctx = new IccContext(outputProfile);
            // outctx.WriteIlluminantRelativeMediaBlackPoint(illuminantRelativeBlackPoint);

            // set output profile description
            outputProfile.HeaderManufacturer = profile.HeaderManufacturer;
            outputProfile.HeaderModel = profile.HeaderModel;
            outputProfile.HeaderAttributes = profile.HeaderAttributes;
            outputProfile.HeaderRenderingIntent = RenderingIntent.ABSOLUTE_COLORIMETRIC;

            var new_desc = $"SDR ACM: {profile.GetInfo(InfoType.Description)}";
            var new_desc_mlu = new MLU(new_desc);
            outputProfile.WriteTag(SafeTagSignature.ProfileDescriptionTag, new_desc_mlu);

            outputProfile.WriteRawTag(MHC2Tag.Signature, mhc2);

            outputProfile.ComputeProfileId();

            return outputProfile;
        }

        public IccProfile CreateIcc(GenerateProfileCommand command)
        {
            var maxNits = command.WhiteLuminance;

            //var wtpt = IlluminantRelativeWhitePoint;

            var devicePrimaries = command.DevicePrimaries; // new RgbPrimaries(new() { x = 0.698, y = 0.292 }, new() { x = 0.255, y = 0.699 }, new() { x = 0.148, y = 0.056 }, new() { x = 0.3127, y = 0.3290 });

            var srgbTrc = profile.ReadTag(SafeTagSignature.RedTRCTag)!;

            var sourcePrimaries = RgbPrimaries.sRGB;

            var xyz_to_srgb = XYZToRgb(sourcePrimaries);


            Matrix<double> user_matrix = DenseMatrix.CreateIdentity(3);

            // pipeline here: input signal converted to XYZ (interpreted as sRGB)

            if (command.ColorGamut != ColorGamut.Native && !ReferenceEquals(sourcePrimaries, RgbPrimaries.sRGB))
            {
                user_matrix = RgbToXYZ(sourcePrimaries) * xyz_to_srgb * user_matrix;
            }

            // pipeline here: input signal converted to XYZ (interpreted as custom RGB)

            if (UseChromaticAdaptation)
            {
                user_matrix = GetChromaticAdaptationMatrix(sourcePrimaries.White.ToXYZ(), devicePrimaries.White.ToXYZ()) * user_matrix;
            }

            // pipeline here: input signal XYZ adapted to device white point

            // hook: scale white point

            //var source_white_to_xyz = user_matrix * new DenseVector(new double[] { 1, 1, 1 });
            //var mapped_y = source_white_to_xyz[1];
            //var profile_max_nits = max_nits * (mapped_y / wtpt.Y);
            var profile_max_nits = maxNits;

            // end hook

            // pipeline here: linear device RGB

            // hack: eliminate fixed sRGB to XYZ transform

            // CSC
            var targetPrimaries = command.ColorGamut switch
            {
                ColorGamut.Native => devicePrimaries,
                ColorGamut.sRGB => RgbPrimaries.sRGB,
                ColorGamut.P3 => RgbPrimaries.P3D65,
                ColorGamut.Rec2020 => RgbPrimaries.Rec2020,
                ColorGamut.AdobeRGB => RgbPrimaries.AdobeRGB,
                ColorGamut.Rec2020Native => CalculateRec2020Native(devicePrimaries),
                _ => devicePrimaries
            };

            {
                var target_to_xyz = RgbToXYZ(targetPrimaries);

                user_matrix = XYZToRgb(devicePrimaries) * user_matrix;

                user_matrix = target_to_xyz * user_matrix;

                // Apply saturation boost via matrix
                // Calculate luminance coefficients from device primaries (Y row of RGB to XYZ matrix)
                var deviceRgbToXyz = RgbToXYZ(devicePrimaries);
                double Lr = deviceRgbToXyz[1, 0]; // Y coefficient for R
                double Lg = deviceRgbToXyz[1, 1]; // Y coefficient for G
                double Lb = deviceRgbToXyz[1, 2]; // Y coefficient for B
                // Normalize so they sum to 1
                double lumSum = Lr + Lg + Lb;
                Lr /= lumSum; Lg /= lumSum; Lb /= lumSum;

                // Saturation boost (gamut expansion)
                double satBoost; // (1.0 = no change, >1 increases saturation)
                if (command.ColorGamut == ColorGamut.Rec2020)
                {
                    satBoost = 0.95;
                }
                else if (command.ColorGamut == ColorGamut.Native)
                {
                    satBoost = 1.05;
                }
                else
                {
                    satBoost = 1.0;
                }

                var saturation_matrix = DenseMatrix.OfArray(new double[,] {
                    { satBoost + (1 - satBoost) * Lr, (1 - satBoost) * Lg, (1 - satBoost) * Lb },
                    { (1 - satBoost) * Lr, satBoost + (1 - satBoost) * Lg, (1 - satBoost) * Lb },
                    { (1 - satBoost) * Lr, (1 - satBoost) * Lg, satBoost + (1 - satBoost) * Lb }
                });
                user_matrix = saturation_matrix * user_matrix;
            }

            var mhc2_matrix = new double[,] {
               { user_matrix[0,0], user_matrix[0,1], user_matrix[0,2], 0 },
               { user_matrix[1,0], user_matrix[1,1], user_matrix[1,2], 0 },
               { user_matrix[2,0], user_matrix[2,1], user_matrix[2,2], 0 },
            };

            MHC2 = new MHC2Tag
            {
                MinCLL = command.MinCLL,
                MaxCLL = command.MaxCLL
            };

            if (command.IsHDRProfile)
            {
                if (command.SDRTransferFunction == SDRTransferFunction.Piecewise)
                {
                    MHC2.ApplyPiecewise(command.SDRBrightnessBoost, command.MinCLL);

                    // Apply S-curve contrast (1.0 = no change, >1 = more contrast)
                    //const double contrastSCurve = 1.25;
                    //MHC2.ApplyContrastSCurve(contrastSCurve);
                }
                else if (command.SDRTransferFunction == SDRTransferFunction.PurePower)
                {
                    MHC2.ApplySdrAcm(command.SDRMaxBrightness, command.SDRMinBrightness, command.Gamma, command.SDRBrightnessBoost, command.ShadowDetailBoost, command.MinCLL);
                }
                else if (command.SDRTransferFunction == SDRTransferFunction.BT_1886)
                {
                    MHC2.ApplySdrAcm(120, command.MinCLL, 2.2, command.SDRBrightnessBoost, command.ShadowDetailBoost, command.MinCLL);

                    // Apply S-curve contrast (1.0 = no change, >1 = more contrast)
                    const double contrastSCurve = 1.2;
                    MHC2.ApplyContrastSCurve(contrastSCurve);
                }
                else if (command.SDRTransferFunction == SDRTransferFunction.ToneMappedPiecewise)
                {

                    double from_nits = command.ToneMappingFromLuminance;
                    double to_nits = command.ToneMappingToLuminance;

                    // Old gamma-PQ hybrid approach (kept for reference):
                    // double numerator = to_nits / from_nits;
                    // double gamma_like = (command.ToneMappingToLuminance / command.HdrGammaMultiplier) / numerator;
                    // double curve_like = (command.ToneMappingToLuminance / command.HdrBrightnessMultiplier) / numerator;
                    // MHC2.ApplyToneMappingCurve(from_nits, to_nits, to_nits);
                    // MHC2.ApplyToneMappingCurve(from_nits, from_nits, curve_like);
                    // MHC2.ApplyToneMappingCurveGamma(from_nits, from_nits, gamma_like);

                    // New PQ-native approach - works entirely in PQ space with smooth toe
                    // Calculate the contrast pivot point (equivalent to old gamma_like)
                    // gamma_like = (to_nits / HdrGammaMultiplier) / (to_nits / from_nits)
                    //            = from_nits / HdrGammaMultiplier
                    double contrastPivot = from_nits / command.HdrGammaMultiplier;

                    // Apply PQ-native tonemapping with brightness adjustment and compression
                    MHC2.ApplyToneMappingCurvePQNative(
                        maxInputNits: from_nits,
                        maxOutputNits: to_nits,
                        contrastPivotNits: contrastPivot,
                        brightnessMultiplier: command.HdrBrightnessMultiplier,
                        maxBrightnessCompression: 1.0,
                        minCLL: command.MinCLL
                    );

                    // Apply S-curve contrast (1.0 = no change, >1 = more contrast)
                    const double contrastSCurve = 1.25;
                    MHC2.ApplyContrastSCurve(contrastSCurve);

                    MHC2.ApplyWOLEDDesaturationCompensation(95);
                }
            }
            else
            {
                if (command.SDRTransferFunction == SDRTransferFunction.PurePower)
                {
                    MHC2.ApplyGamma(command.Gamma, command.ShadowDetailBoost);
                }
                else
                {
                    MHC2.ApplyPiecewise(command.SDRBrightnessBoost, command.MinCLL);
                }
            }

            MHC2.Matrix3x4 = mhc2_matrix;

            var outputProfile = IccProfile.CreateRGB(devicePrimaries.White.ToXYZ().ToCIExyY(), new CIExyYTRIPLE
            {
                Red = devicePrimaries.Red.ToXYZ().ToCIExyY(),
                Green = devicePrimaries.Green.ToXYZ().ToCIExyY(),
                Blue = devicePrimaries.Blue.ToXYZ().ToCIExyY()
            }, new RgbToneCurve(srgbTrc, srgbTrc, srgbTrc));

            outputProfile.WriteTag(SafeTagSignature.LuminanceTag, new CIEXYZ { Y = profile_max_nits });

            var outctx = new IccContext(outputProfile);
            outctx.WriteIlluminantRelativeMediaBlackPoint(illuminantRelativeBlackPoint);

            // copy device description from device profile
            var copy_tags = new TagSignature[] { TagSignature.DeviceMfgDescTag, TagSignature.DeviceModelDescTag };

            unsafe
            {
                foreach (var tag in copy_tags)
                {
                    var tag_ptr = profile.ReadTag(tag);
                    if (tag_ptr != null)
                    {
                        outputProfile.WriteTag(tag, tag_ptr);
                    }
                }
            }

            // set output profile description
            outputProfile.HeaderManufacturer = profile.HeaderManufacturer;
            outputProfile.HeaderModel = profile.HeaderModel;
            outputProfile.HeaderAttributes = profile.HeaderAttributes;
            outputProfile.HeaderRenderingIntent = RenderingIntent.PERCEPTUAL;

            var descAppendix = $" ({GetDeviceDescription()})";
            var new_desc = command.Description ?? "";
            if (!new_desc.Contains(descAppendix))
            {
                new_desc += descAppendix;
            }
            var new_desc_mlu = new MLU(new_desc);
            outputProfile.WriteTag(SafeTagSignature.ProfileDescriptionTag, new_desc_mlu);

            var extraInfoTag = new ExtraInfoTag
            {
                SDRTransferFunction = command.SDRTransferFunction,
                Gamma = command.Gamma,
                SDRMinBrightness = command.SDRMinBrightness,
                SDRMaxBrightness = command.SDRMaxBrightness,
                SDRBrightnessBoost = command.SDRBrightnessBoost,
                ShadowDetailBoost = command.ShadowDetailBoost,
                TargetGamut = command.ColorGamut,
                ToneMappingFromLuminance = command.ToneMappingFromLuminance,
                ToneMappingToLuminance = command.ToneMappingToLuminance,
                HdrGammaMultiplier = command.HdrGammaMultiplier,
                HdrBrightnessMultiplier = command.HdrBrightnessMultiplier,
                WOLEDDesaturationCompensation = 95
            };

            var ccDesc = JsonSerializer.Serialize(extraInfoTag);

            outputProfile.WriteTag(SafeTagSignature.ScreeningDescTag, new MLU(ccDesc));

            outputProfile.WriteRawTag(MHC2Tag.Signature, MHC2.ToBytes());

            outputProfile.ComputeProfileId();

            return outputProfile;
        }

        public IccProfile CreateCscIcc(RgbPrimaries? sourcePrimaries = null, string sourceDescription = "sRGB")
        {
            var wtpt = IlluminantRelativeWhitePoint;
            var vcgt = profile.ReadTagOrDefault(SafeTagSignature.VcgtTag)?.ToArray();

            var customPrimaries = new RgbPrimaries(new() { x = 0.698, y = 0.292 }, new() { x = 0.255, y = 0.699 }, new() { x = 0.148, y = 0.056 }, new() { x = 0.3127, y = 0.3290 });

            var devicePrimaries = customPrimaries;

            var deviceOetf = new ToneCurve[] { profileRedReverseToneCurve, profileGreenReverseToneCurve, profileBlueReverseToneCurve };

            var srgbTrc = IccProfile.Create_sRGB().ReadTag(SafeTagSignature.RedTRCTag)!;
            var sourceEotf = new ToneCurve[] { srgbTrc, srgbTrc, srgbTrc };

            sourcePrimaries ??= RgbPrimaries.sRGB;

            var srgb_to_xyz = RgbToXYZ(RgbPrimaries.sRGB);
            var xyz_to_srgb = XYZToRgb(RgbPrimaries.sRGB);


            Matrix<double> user_matrix = DenseMatrix.CreateIdentity(3);

            // pipeline here: input signal converted to XYZ (interpreted as sRGB)

            if (!ReferenceEquals(sourcePrimaries, RgbPrimaries.sRGB))
            {
                user_matrix = RgbToXYZ(sourcePrimaries) * xyz_to_srgb * user_matrix;
            }

            // pipeline here: input signal converted to XYZ (interpreted as custom RGB)

            if (UseChromaticAdaptation)
            {
                user_matrix = GetChromaticAdaptationMatrix(sourcePrimaries.White.ToXYZ(), devicePrimaries.White.ToXYZ()) * user_matrix;
            }

            // pipeline here: input signal XYZ adapted to device white point

            // hook: scale white point

            var source_white_to_xyz = user_matrix * new DenseVector(new double[] { 1, 1, 1 });
            var mapped_y = source_white_to_xyz[1];
            var profile_max_nits = max_nits * (mapped_y / wtpt.Y);

            // end hook

            user_matrix = XYZToRgb(devicePrimaries) * user_matrix;

            // pipeline here: linear device RGB

            // hack: eliminate fixed sRGB to XYZ transform

            var mhc2_matrix = new double[,] {
               { user_matrix[0,0], user_matrix[0,1], user_matrix[0,2], 0 },
               { user_matrix[1,0], user_matrix[1,1], user_matrix[1,2], 0 },
               { user_matrix[2,0], user_matrix[2,1], user_matrix[2,2], 0 },
            };

            double[,] mhc2_lut;
            if (vcgt != null)
            {
                var lut_size = 1024;
                mhc2_lut = new double[3, lut_size];
                for (int ch = 0; ch < 3; ch++)
                {
                    for (int iinput = 0; iinput < lut_size; iinput++)
                    {
                        var input = (float)iinput / (lut_size - 1);
                        var linear = sourceEotf[ch].EvalF32(input);
                        var dev_output = deviceOetf[ch].EvalF32(linear);
                        if (vcgt != null)
                        {
                            dev_output = vcgt[ch].EvalF32(dev_output);
                        }
                        mhc2_lut[ch, iinput] = dev_output;
                    }
                }
            }
            else
            {
                var lut_size = 1024;
                mhc2_lut = new double[3, lut_size];
                for (int ch = 0; ch < 3; ch++)
                {
                    for (int iinput = 0; iinput < lut_size; iinput++)
                    {
                        var input = (float)iinput / (lut_size - 1);
                        var linear = sourceEotf[ch].EvalF32(input);

                        var dev_output = CmsFunctions.RgbToLinear(linear, 1.6);

                        //var dev_output = deviceOetf[ch].EvalF32(input);
                        //if (vcgt != null)
                        //{
                        //    dev_output = vcgt[ch].EvalF32(dev_output);
                        //}
                        mhc2_lut[ch, iinput] = dev_output;
                    }
                }
            }

            var mhc2d = new MHC2Tag
            {
                MinCLL = min_nits,
                MaxCLL = max_nits,
                Matrix3x4 = mhc2_matrix,
                RegammaLUT = mhc2_lut
            };

            var mhc2 = mhc2d.ToBytes();

            var outputProfile = IccProfile.CreateRGB(sourcePrimaries.White.ToXYZ().ToCIExyY(), new CIExyYTRIPLE
            {
                Red = sourcePrimaries.Red.ToXYZ().ToCIExyY(),
                Green = sourcePrimaries.Green.ToXYZ().ToCIExyY(),
                Blue = sourcePrimaries.Blue.ToXYZ().ToCIExyY()
            }, new RgbToneCurve(srgbTrc, srgbTrc, srgbTrc));

            outputProfile.WriteTag(SafeTagSignature.LuminanceTag, new CIEXYZ { Y = profile_max_nits });

            var outctx = new IccContext(outputProfile);
            outctx.WriteIlluminantRelativeMediaBlackPoint(illuminantRelativeBlackPoint);

            // copy device description from device profile
            var copy_tags = new TagSignature[] { TagSignature.DeviceMfgDescTag, TagSignature.DeviceModelDescTag };

            unsafe
            {
                foreach (var tag in copy_tags)
                {
                    var tag_ptr = profile.ReadTag(tag);
                    if (tag_ptr != null)
                    {
                        outputProfile.WriteTag(tag, tag_ptr);
                    }
                }
            }

            // set output profile description
            outputProfile.HeaderManufacturer = profile.HeaderManufacturer;
            outputProfile.HeaderModel = profile.HeaderModel;
            outputProfile.HeaderAttributes = profile.HeaderAttributes;
            outputProfile.HeaderRenderingIntent = RenderingIntent.PERCEPTUAL;

            var new_desc = $"XCSC: {sourceDescription} ({GetDeviceDescription()})";
            var new_desc_mlu = new MLU(new_desc);
            outputProfile.WriteTag(SafeTagSignature.ProfileDescriptionTag, new_desc_mlu);

            outputProfile.WriteRawTag(MHC2Tag.Signature, mhc2);

            outputProfile.ComputeProfileId();

            return outputProfile;
        }

    }
}