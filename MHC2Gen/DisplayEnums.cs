using System.ComponentModel;

namespace MHC2Gen;

public enum SaveOption
{
    Install,
    [Description("Install and associate to display")]
    InstallAndAssociate,
    [Description("Install and set as display default")]
    InstallAndSetAsDefault,
    [Description("Save to file")]
    SaveToFile
}

public enum DisplayPrimariesSource
{
    Custom,
    EDID,
    Windows,
    ColorProfile,
}

public enum SDRTransferFunction
{
    [Description("Corrected Gamma 2.2 Piecewise")]
    CorrectedGamma22Piecewise = 0,
    [Description("Piecewise")]
    Piecewise = 2,
}

public enum ColorGamut
{
    Native = 0,
    sRGB = 1,
    P3 = 2,
    Rec2020 = 3,
    AdobeRGB = 4,
    [Description("Rec2020 (Native Adapted)")]
    Rec2020Native = 5
}
