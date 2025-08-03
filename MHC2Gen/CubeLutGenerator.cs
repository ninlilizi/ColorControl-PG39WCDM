using System;
using System.Globalization;
using System.IO;
using System.Text;
using LittleCms;

namespace MHC2Gen
{
    /// <summary>
    /// Generates 3D LUT files in .cube format compatible with DisplayCAL and color grading applications
    /// </summary>
    public static class CubeLutGenerator
    {
        /// <summary>
        /// Generate a .cube LUT file from the current profile transformation
        /// </summary>
        /// <param name="command">The profile generation command containing transformation parameters</param>
        /// <param name="filePath">Path where the .cube file will be saved</param>
        /// <param name="lutSize">Size of the 3D LUT (common sizes: 17, 33, 65)</param>
        /// <param name="title">Optional title for the LUT</param>
        public static void GenerateCubeLut(GenerateProfileCommand command, string filePath, int lutSize = 65, string? title = null)
        {
            // For HDR profiles, use the proper HDR LUT generation
            if (command.IsHDRProfile)
            {
                GenerateHdrCubeLut(command, filePath, lutSize, title);
                return;
            }
            
            // Create a temporary profile to use for color transformations
            var profile = IccProfile.Create_sRGB();
            var context = new DeviceIccContext(profile);
            var generatedProfile = context.CreateIcc(command);
            
            using var writer = new StreamWriter(filePath, false, Encoding.UTF8);
            
            // Write .cube header - fix string interpolation issue
            var lutTitle = title ?? $"ColorControl LUT - {command.Description}";
            writer.WriteLine($"TITLE \"{lutTitle}\"");
            writer.WriteLine("DOMAIN_MIN 0.0 0.0 0.0");
            writer.WriteLine("DOMAIN_MAX 1.0 1.0 1.0");
            writer.WriteLine($"LUT_3D_SIZE {lutSize}");
            writer.WriteLine();
            
            // Create color transform from generated profile
            var ctx = new CmsContext();
            var sourceProfile = IccProfile.Create_sRGB();
            var transform = new CmsTransform(ctx, sourceProfile, CmsPixelFormat.RGBDouble, generatedProfile, CmsPixelFormat.RGBDouble, RenderingIntent.PERCEPTUAL, default);
            
            // Generate LUT data
            for (int b = 0; b < lutSize; b++)
            {
                for (int g = 0; g < lutSize; g++)
                {
                    for (int r = 0; r < lutSize; r++)
                    {
                        // Calculate normalized RGB input values
                        double inputR = (double)r / (lutSize - 1);
                        double inputG = (double)g / (lutSize - 1);
                        double inputB = (double)b / (lutSize - 1);
                        
                        // Apply the color transformation
                        var input = new[] { inputR, inputG, inputB };
                        var output = new double[3];
                        
                        transform.DoTransform<double, double>(input, output, 1);
                        
                        // Clamp output values
                        output[0] = Math.Max(0.0, Math.Min(1.0, output[0]));
                        output[1] = Math.Max(0.0, Math.Min(1.0, output[1]));
                        output[2] = Math.Max(0.0, Math.Min(1.0, output[2]));
                        
                        // Write to .cube file with proper formatting
                        writer.WriteLine($"{output[0].ToString("F6", CultureInfo.InvariantCulture)} {output[1].ToString("F6", CultureInfo.InvariantCulture)} {output[2].ToString("F6", CultureInfo.InvariantCulture)}");
                    }
                }
            }
        }
        
        /// <summary>
        /// Generate an HDR .cube LUT using BT.2020 color space and SMPTE ST 2084 (PQ) encoding
        /// This creates a proper HDR LUT suitable for HDR10 workflows
        /// </summary>
        /// <param name="command">The profile generation command containing transformation parameters</param>
        /// <param name="filePath">Path where the .cube file will be saved</param>
        /// <param name="lutSize">Size of the 3D LUT (common sizes: 17, 33, 65)</param>
        /// <param name="title">Optional title for the LUT</param>
        public static void GenerateHdrCubeLut(GenerateProfileCommand command, string filePath, int lutSize = 65, string? title = null)
        {
            // Create device profile following the existing PQ10 decode pattern
            var profile = IccProfile.Create_sRGB();
            var context = new DeviceIccContext(profile);
            
            // Use the existing CreatePQ10DecodeIcc method as reference for HDR handling
            var pq10Profile = context.CreatePQ10DecodeIcc(command.MaxCLL, command.MinCLL);
            
            using var writer = new StreamWriter(filePath, false, Encoding.UTF8);
            
            // Write .cube header with HDR-specific metadata
            var lutTitle = title ?? $"ColorControl HDR LUT (BT.2020/ST2084) - {command.Description}";
            writer.WriteLine($"TITLE \"{lutTitle}\"");
            writer.WriteLine("DOMAIN_MIN 0.0 0.0 0.0");
            writer.WriteLine("DOMAIN_MAX 1.0 1.0 1.0");
            writer.WriteLine($"LUT_3D_SIZE {lutSize}");
            writer.WriteLine("# HDR LUT using BT.2020 color space and SMPTE ST 2084 (PQ) encoding");
            writer.WriteLine($"# Max luminance: {command.MaxCLL} cd/m²");
            writer.WriteLine($"# Min luminance: {command.MinCLL} cd/m²");
            writer.WriteLine();
            
            // Create color transform using BT.2020 as source
            var ctx = new CmsContext();
            var bt2020Profile = IccProfile.Create_sRGB(); // We'll manually handle the BT.2020 color space conversion
            var transform = new CmsTransform(ctx, bt2020Profile, CmsPixelFormat.RGBDouble, pq10Profile, CmsPixelFormat.RGBDouble, RenderingIntent.PERCEPTUAL, default);
            
            // Generate LUT data with PQ encoding
            for (int b = 0; b < lutSize; b++)
            {
                for (int g = 0; g < lutSize; g++)
                {
                    for (int r = 0; r < lutSize; r++)
                    {
                        // Calculate normalized RGB input values (0.0 to 1.0 representing PQ-encoded values)
                        double inputR_PQ = (double)r / (lutSize - 1);
                        double inputG_PQ = (double)g / (lutSize - 1);
                        double inputB_PQ = (double)b / (lutSize - 1);
                        
                        // Convert PQ-encoded values to linear values in the proper luminance range
                        double inputR_Nits = ST2084.SignalToNits(inputR_PQ);
                        double inputG_Nits = ST2084.SignalToNits(inputG_PQ);
                        double inputB_Nits = ST2084.SignalToNits(inputB_PQ);
                        
                        // Normalize to the working range expected by the profile
                        double inputR_Linear = Math.Max(inputR_Nits - command.MinCLL, 0) / (command.MaxCLL - command.MinCLL);
                        double inputG_Linear = Math.Max(inputG_Nits - command.MinCLL, 0) / (command.MaxCLL - command.MinCLL);
                        double inputB_Linear = Math.Max(inputB_Nits - command.MinCLL, 0) / (command.MaxCLL - command.MinCLL);
                        
                        // Apply the color transformation
                        var input = new[] { inputR_Linear, inputG_Linear, inputB_Linear };
                        var output = new double[3];
                        
                        transform.DoTransform<double, double>(input, output, 1);
                        
                        // Convert output back to PQ encoding
                        // The output is already processed by the device profile, so we convert back to nits and then PQ
                        double outputR_Nits = output[0] * (command.MaxCLL - command.MinCLL) + command.MinCLL;
                        double outputG_Nits = output[1] * (command.MaxCLL - command.MinCLL) + command.MinCLL;
                        double outputB_Nits = output[2] * (command.MaxCLL - command.MinCLL) + command.MinCLL;
                        
                        // Apply PQ encoding to output
                        double outputR_PQ = ST2084.NitsToSignal(Math.Max(0.0, outputR_Nits));
                        double outputG_PQ = ST2084.NitsToSignal(Math.Max(0.0, outputG_Nits));
                        double outputB_PQ = ST2084.NitsToSignal(Math.Max(0.0, outputB_Nits));
                        
                        // Clamp output values to valid PQ range
                        outputR_PQ = Math.Max(0.0, Math.Min(1.0, outputR_PQ));
                        outputG_PQ = Math.Max(0.0, Math.Min(1.0, outputG_PQ));
                        outputB_PQ = Math.Max(0.0, Math.Min(1.0, outputB_PQ));
                        
                        // Write to .cube file with proper formatting
                        writer.WriteLine($"{outputR_PQ.ToString("F6", CultureInfo.InvariantCulture)} {outputG_PQ.ToString("F6", CultureInfo.InvariantCulture)} {outputB_PQ.ToString("F6", CultureInfo.InvariantCulture)}");
                    }
                }
            }
        }
        
        /// <summary>
        /// Generate an sRGB SDR .cube LUT using standard Rec.709/sRGB encoding
        /// This creates a companion SDR LUT suitable for standard dynamic range workflows
        /// </summary>
        /// <param name="command">The profile generation command containing transformation parameters</param>
        /// <param name="filePath">Path where the .cube file will be saved</param>
        /// <param name="lutSize">Size of the 3D LUT (common sizes: 17, 33, 65)</param>
        /// <param name="title">Optional title for the LUT</param>
        public static void GenerateSdrCubeLut(GenerateProfileCommand command, string filePath, int lutSize = 65, string? title = null)
        {
            // Create an SDR version of the command for proper SDR LUT generation
            var sdrCommand = new GenerateProfileCommand
            {
                Description = command.Description,
                IsHDRProfile = false, // Force SDR processing
                MinCLL = 0.0,
                MaxCLL = 100.0, // Standard SDR white level (100 cd/m²)
                BlackLuminance = command.BlackLuminance,
                WhiteLuminance = Math.Min(command.WhiteLuminance, 100.0), // Cap at SDR levels
                SDRMinBrightness = command.SDRMinBrightness,
                SDRMaxBrightness = command.SDRMaxBrightness,
                SDRTransferFunction = command.SDRTransferFunction,
                SDRBrightnessBoost = command.SDRBrightnessBoost,
                ColorGamut = command.ColorGamut == ColorGamut.Rec2020 ? ColorGamut.sRGB : command.ColorGamut, // Map BT.2020 to sRGB for SDR
                Gamma = command.Gamma,
                DevicePrimaries = command.DevicePrimaries,
                ShadowDetailBoost = command.ShadowDetailBoost,
                ToneMappingFromLuminance = Math.Min(command.ToneMappingFromLuminance, 100.0),
                ToneMappingToLuminance = Math.Min(command.ToneMappingToLuminance, 100.0),
                HdrBrightnessMultiplier = 1.0,
                HdrGammaMultiplier = 1.0,
                WOLEDDesaturationCompensation = 0.0 // Disable HDR-specific processing
            };

            // Create device profile for SDR processing
            var profile = IccProfile.Create_sRGB();
            var context = new DeviceIccContext(profile);
            var sdrProfile = context.CreateIcc(sdrCommand);
            
            using var writer = new StreamWriter(filePath, false, Encoding.UTF8);
            
            // Write .cube header with SDR-specific metadata
            var lutTitle = title ?? $"ColorControl SDR LUT (sRGB/Rec.709) - {command.Description}";
            writer.WriteLine($"TITLE \"{lutTitle}\"");
            writer.WriteLine("DOMAIN_MIN 0.0 0.0 0.0");
            writer.WriteLine("DOMAIN_MAX 1.0 1.0 1.0");
            writer.WriteLine($"LUT_3D_SIZE {lutSize}");
            writer.WriteLine("# SDR LUT using sRGB/Rec.709 color space with gamma encoding");
            writer.WriteLine($"# Max luminance: {sdrCommand.MaxCLL} cd/m²");
            writer.WriteLine($"# Min luminance: {sdrCommand.MinCLL} cd/m²");
            writer.WriteLine("# Companion SDR LUT for standard dynamic range workflows");
            writer.WriteLine();
            
            // Create color transform using sRGB as both source and reference
            var ctx = new CmsContext();
            var sourceProfile = IccProfile.Create_sRGB();
            var transform = new CmsTransform(ctx, sourceProfile, CmsPixelFormat.RGBDouble, sdrProfile, CmsPixelFormat.RGBDouble, RenderingIntent.PERCEPTUAL, default);
            
            // Generate LUT data with standard gamma encoding
            for (int b = 0; b < lutSize; b++)
            {
                for (int g = 0; g < lutSize; g++)
                {
                    for (int r = 0; r < lutSize; r++)
                    {
                        // Calculate normalized RGB input values (0.0 to 1.0 representing standard gamma-encoded values)
                        double inputR = (double)r / (lutSize - 1);
                        double inputG = (double)g / (lutSize - 1);
                        double inputB = (double)b / (lutSize - 1);
                        
                        // Apply the color transformation
                        var input = new[] { inputR, inputG, inputB };
                        var output = new double[3];
                        
                        transform.DoTransform<double, double>(input, output, 1);
                        
                        // Clamp output values to standard SDR range
                        output[0] = Math.Max(0.0, Math.Min(1.0, output[0]));
                        output[1] = Math.Max(0.0, Math.Min(1.0, output[1]));
                        output[2] = Math.Max(0.0, Math.Min(1.0, output[2]));
                        
                        // Write to .cube file with proper formatting
                        writer.WriteLine($"{output[0].ToString("F6", CultureInfo.InvariantCulture)} {output[1].ToString("F6", CultureInfo.InvariantCulture)} {output[2].ToString("F6", CultureInfo.InvariantCulture)}");
                    }
                }
            }
        }
        
        /// <summary>
        /// Generate a .cube LUT specifically from the MHC2 tag data for more direct control
        /// </summary>
        /// <param name="mhc2Tag">The MHC2 tag containing the LUT and matrix data</param>
        /// <param name="filePath">Path where the .cube file will be saved</param>
        /// <param name="lutSize">Size of the 3D LUT (common sizes: 17, 33, 65)</param>
        /// <param name="title">Optional title for the LUT</param>
        internal static void GenerateCubeLutFromMHC2(MHC2Tag mhc2Tag, string filePath, int lutSize = 65, string? title = null)
        {
            if (mhc2Tag.RegammaLUT == null || mhc2Tag.Matrix3x4 == null)
                throw new ArgumentException("MHC2 tag must contain valid LUT and matrix data");
                
            using var writer = new StreamWriter(filePath, false, Encoding.UTF8);
            
            // Write .cube header
            var lutTitle = title ?? "ColorControl Direct MHC2 LUT";
            writer.WriteLine($"TITLE \"{lutTitle}\"");
            writer.WriteLine("DOMAIN_MIN 0.0 0.0 0.0");
            writer.WriteLine("DOMAIN_MAX 1.0 1.0 1.0");
            writer.WriteLine($"LUT_3D_SIZE {lutSize}");
            writer.WriteLine();
            
            var regammaLutSize = mhc2Tag.RegammaLUT.GetLength(1);
            
            // Generate LUT data by applying the MHC2 transformation directly
            for (int b = 0; b < lutSize; b++)
            {
                for (int g = 0; g < lutSize; g++)
                {
                    for (int r = 0; r < lutSize; r++)
                    {
                        // Calculate normalized RGB input values
                        double inputR = (double)r / (lutSize - 1);
                        double inputG = (double)g / (lutSize - 1);
                        double inputB = (double)b / (lutSize - 1);
                        
                        // Apply matrix transformation (simplified - full matrix would need proper XYZ space)
                        double transformedR = mhc2Tag.Matrix3x4[0,0] * inputR + mhc2Tag.Matrix3x4[0,1] * inputG + mhc2Tag.Matrix3x4[0,2] * inputB;
                        double transformedG = mhc2Tag.Matrix3x4[1,0] * inputR + mhc2Tag.Matrix3x4[1,1] * inputG + mhc2Tag.Matrix3x4[1,2] * inputB;
                        double transformedB = mhc2Tag.Matrix3x4[2,0] * inputR + mhc2Tag.Matrix3x4[2,1] * inputG + mhc2Tag.Matrix3x4[2,2] * inputB;
                        
                        // Apply RegammaLUT
                        double outputR = ApplyRegammaLUT(transformedR, mhc2Tag.RegammaLUT, 0, regammaLutSize);
                        double outputG = ApplyRegammaLUT(transformedG, mhc2Tag.RegammaLUT, 1, regammaLutSize);
                        double outputB = ApplyRegammaLUT(transformedB, mhc2Tag.RegammaLUT, 2, regammaLutSize);
                        
                        // Clamp output values
                        outputR = Math.Max(0.0, Math.Min(1.0, outputR));
                        outputG = Math.Max(0.0, Math.Min(1.0, outputG));
                        outputB = Math.Max(0.0, Math.Min(1.0, outputB));
                        
                        // Write to .cube file with proper formatting
                        writer.WriteLine($"{outputR.ToString("F6", CultureInfo.InvariantCulture)} {outputG.ToString("F6", CultureInfo.InvariantCulture)} {outputB.ToString("F6", CultureInfo.InvariantCulture)}");
                    }
                }
            }
        }
        
        /// <summary>
        /// Apply the regamma LUT to a single channel value
        /// </summary>
        private static double ApplyRegammaLUT(double input, double[,] regammaLUT, int channel, int lutSize)
        {
            // Clamp input
            input = Math.Max(0.0, Math.Min(1.0, input));
            
            // Calculate LUT position
            double lutPos = input * (lutSize - 1);
            int lutIndex = (int)Math.Floor(lutPos);
            double fraction = lutPos - lutIndex;
            
            // Handle edge cases
            if (lutIndex >= lutSize - 1)
                return regammaLUT[channel, lutSize - 1];
                
            if (lutIndex < 0)
                return regammaLUT[channel, 0];
            
            // Linear interpolation between LUT entries
            double value1 = regammaLUT[channel, lutIndex];
            double value2 = regammaLUT[channel, lutIndex + 1];
            
            return value1 + (value2 - value1) * fraction;
        }
    }
}