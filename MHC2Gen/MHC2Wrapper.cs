using LittleCms;
using System;
using System.IO;

namespace MHC2Gen;

public class MHC2Wrapper
{
    public enum NamedGamut
    {
        sRGB,
        AdobeRGB,
        P3D65,
        BT2020
    }

    public static byte[] GenerateSdrAcmProfile(GenerateProfileCommand command)
    {
        var profile = IccProfile.Create_sRGB();

        var context = new DeviceIccContext(profile);

        var newProfile = context.CreateIcc(command);

        return newProfile.GetBytes();

    }

    /// <summary>
    /// Generate a .cube LUT file from a profile generation command
    /// </summary>
    /// <param name="command">Profile generation command</param>
    /// <param name="filePath">Path where the .cube file will be saved</param>
    /// <param name="lutSize">Size of the 3D LUT (17, 33, 65, etc.)</param>
    /// <param name="title">Optional title for the LUT</param>
    public static void GenerateCubeLutFile(GenerateProfileCommand command, string filePath, int lutSize = 65, string? title = null)
    {
        CubeLutGenerator.GenerateCubeLut(command, filePath, lutSize, title);
    }

    /// <summary>
    /// Generate an HDR .cube LUT file using BT.2020 color space and SMPTE ST 2084 (PQ) encoding
    /// For HDR workflows requiring proper HDR10 LUT format
    /// </summary>
    /// <param name="command">Profile generation command</param>
    /// <param name="filePath">Path where the .cube file will be saved</param>
    /// <param name="lutSize">Size of the 3D LUT (17, 33, 65, etc.)</param>
    /// <param name="title">Optional title for the LUT</param>
    public static void GenerateHdrCubeLutFile(GenerateProfileCommand command, string filePath, int lutSize = 65, string? title = null)
    {
        CubeLutGenerator.GenerateHdrCubeLut(command, filePath, lutSize, title);
    }

    /// <summary>
    /// Generate an sRGB SDR .cube LUT file using standard Rec.709/sRGB encoding
    /// For standard dynamic range workflows as a companion to HDR LUTs
    /// </summary>
    /// <param name="command">Profile generation command</param>
    /// <param name="filePath">Path where the .cube file will be saved</param>
    /// <param name="lutSize">Size of the 3D LUT (17, 33, 65, etc.)</param>
    /// <param name="title">Optional title for the LUT</param>
    public static void GenerateSdrCubeLutFile(GenerateProfileCommand command, string filePath, int lutSize = 65, string? title = null)
    {
        CubeLutGenerator.GenerateSdrCubeLut(command, filePath, lutSize, title);
    }

    /// <summary>
    /// Generate both HDR and SDR .cube LUT files for comprehensive workflow support
    /// </summary>
    /// <param name="command">Profile generation command</param>
    /// <param name="hdrFilePath">Path where the HDR .cube file will be saved</param>
    /// <param name="sdrFilePath">Path where the SDR .cube file will be saved</param>
    /// <param name="lutSize">Size of the 3D LUT (17, 33, 65, etc.)</param>
    /// <param name="title">Optional title for the LUTs</param>
    public static void GenerateDualCubeLutFiles(GenerateProfileCommand command, string hdrFilePath, string sdrFilePath, int lutSize = 65, string? title = null)
    {
        CubeLutGenerator.GenerateHdrCubeLut(command, hdrFilePath, lutSize, title != null ? $"{title} - HDR" : null);
        CubeLutGenerator.GenerateSdrCubeLut(command, sdrFilePath, lutSize, title != null ? $"{title} - SDR" : null);
    }

    /// <summary>
    /// Generate both ICC profile and .cube LUT file
    /// </summary>
    /// <param name="command">Profile generation command</param>
    /// <param name="iccFilePath">Path for the ICC profile</param>
    /// <param name="cubeFilePath">Path for the .cube LUT file</param>
    /// <param name="lutSize">Size of the 3D LUT</param>
    /// <param name="title">Optional title for the LUT</param>
    /// <returns>ICC profile bytes</returns>
    public static byte[] GenerateProfileAndCubeLut(GenerateProfileCommand command, string iccFilePath, string cubeFilePath, int lutSize = 65, string? title = null)
    {
        var profileBytes = GenerateSdrAcmProfile(command);
        
        // Save ICC profile
        File.WriteAllBytes(iccFilePath, profileBytes);
        
        // Generate .cube LUT
        CubeLutGenerator.GenerateCubeLut(command, cubeFilePath, lutSize, title);
        
        return profileBytes;
    }

    public static (double MinNits, double MaxNits) GetMinMaxLuminance(string profileName)
    {
        var fileName = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), $@"System32\spool\drivers\color\{profileName}");

        var bytes = File.ReadAllBytes(fileName);

        var profile = IccProfile.Open(bytes.AsSpan());

        var deviceContext = new DeviceIccContext(profile);

        return (deviceContext.min_nits, deviceContext.max_nits);
    }

    public static GenerateProfileCommand LoadProfile(string fileName, bool isAssociatedAsHdr)
    {
        if (fileName.IndexOf("\\") == -1 || !File.Exists(fileName))
        {
            fileName = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), $@"System32\spool\drivers\color\{fileName}");
        }

        var bytes = File.ReadAllBytes(fileName);

        var profile = IccProfile.Open(bytes.AsSpan());

        var deviceContext = new DeviceIccContext(profile);

        return new GenerateProfileCommand
        {
            Description = deviceContext.GetDescription(),
            HasExtraInfo = deviceContext.ExtraInfoTag != null,
            IsHDRProfile = isAssociatedAsHdr,
            BlackLuminance = deviceContext.min_nits,
            WhiteLuminance = deviceContext.max_nits,
            ColorGamut = deviceContext.ExtraInfoTag?.TargetGamut ?? ColorGamut.Native,
            DevicePrimaries = new RgbPrimaries(deviceContext.ProfilePrimaries.Red, deviceContext.ProfilePrimaries.Green, deviceContext.ProfilePrimaries.Blue, deviceContext.ProfilePrimaries.White),
            MinCLL = deviceContext.MHC2?.MinCLL ?? deviceContext.min_nits,
            MaxCLL = deviceContext.MHC2?.MaxCLL ?? deviceContext.max_nits,
            SDRMinBrightness = deviceContext.ExtraInfoTag?.SDRMinBrightness ?? 0,
            SDRMaxBrightness = deviceContext.ExtraInfoTag?.SDRMaxBrightness ?? 100,
            SDRTransferFunction = deviceContext.ExtraInfoTag?.SDRTransferFunction ?? SDRTransferFunction.ToneMappedPiecewise,
            SDRBrightnessBoost = deviceContext.ExtraInfoTag?.SDRBrightnessBoost ?? 0,
            ShadowDetailBoost = deviceContext.ExtraInfoTag?.ShadowDetailBoost ?? 0,
            Gamma = deviceContext.ExtraInfoTag?.Gamma ?? 2.4,
            ToneMappingFromLuminance = deviceContext.ExtraInfoTag?.ToneMappingFromLuminance ?? 400,
            ToneMappingToLuminance = deviceContext.ExtraInfoTag?.ToneMappingToLuminance ?? 400,
            HdrGammaMultiplier = deviceContext.ExtraInfoTag?.HdrGammaMultiplier ?? 1,
            HdrBrightnessMultiplier = deviceContext.ExtraInfoTag?.HdrBrightnessMultiplier ?? 1,
            WOLEDDesaturationCompensation = deviceContext.ExtraInfoTag?.WOLEDDesaturationCompensation ?? 0
        };
    }
}
