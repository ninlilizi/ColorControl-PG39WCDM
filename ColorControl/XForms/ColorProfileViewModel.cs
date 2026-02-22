using ColorControl.Services.Common;
using ColorControl.Shared.Common;
using ColorControl.Shared.Forms;
using ColorControl.Shared.Native;
using ColorControl.Shared.XForms;
using MHC2Gen;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Windows;
using System.Windows.Forms;
using WindowsDisplayAPI;

namespace ColorControl.XForms;

public class CustomDisplay
{
	public Display Display { get; set; }

	public string DisplayName => Display.DisplayName;
	public string DevicePath => Display.DevicePath;
	public bool IsHDRSupportedAndEnabled()
	{
		var advancedColor = CCD.GetAdvancedColorSupportedAndEnabled(DisplayName);

		return advancedColor.isSupported && advancedColor.isEnabled;
	}

	public CustomDisplay(Display display)
	{
		Display = display;
	}

	public override string ToString()
	{
		return $"{CCD.GetDisplayInfo(Display.DisplayName)} on {Display.Adapter.DeviceName}";
	}
}

public class ColorPointModel : BaseViewModel
{
	private double _x;
	public double X { get => _x; set { _x = value; OnPropertyChanged(); } }

	private double _y;
	public double Y { get => _y; set { _y = value; OnPropertyChanged(); } }

	public ColorPointModel()
	{

	}

	public ColorPointModel(CIExy coord)
	{
		X = coord.x;
		Y = coord.y;
	}
}

internal class ColorProfileViewModel : BaseViewModel
{
	[StringLength(30)]
	public string Description { get; set; }

	public ColorPointModel RedPoint { get; set; } = new ColorPointModel(RgbPrimaries.sRGB.Red);
	public ColorPointModel GreenPoint { get; set; } = new ColorPointModel(RgbPrimaries.sRGB.Green);
	public ColorPointModel BluePoint { get; set; } = new ColorPointModel(RgbPrimaries.sRGB.Blue);
	public ColorPointModel WhitePoint { get; set; } = new ColorPointModel(RgbPrimaries.sRGB.White);
	public Dictionary<DisplayPrimariesSource, string> PrimariesSources { get; } = Utils.EnumToDictionary<DisplayPrimariesSource>();
	public DisplayPrimariesSource PrimariesSource { get; set; } = DisplayPrimariesSource.EDID;
	public Dictionary<ColorGamut, string> ColorGamuts { get; } = Utils.EnumToDictionary<ColorGamut>();
	public ColorGamut ColorGamut { get; set; } = ColorGamut.Rec2020Native;
	public Dictionary<SDRTransferFunction, string> SDRTransferFunctions { get; } = Utils.EnumToDictionary<SDRTransferFunction>();
	public SDRTransferFunction SDRTransferFunction { get; set; } = SDRTransferFunction.CorrectedGamma22Piecewise;
	[Range(0.1, 10)]
	public double CustomGamma { get; set; } = 2.20;
	[Range(0, 10)]
	public double MinCLL { get; set; } = 0.00248; // Min panel can display
	// public double MinCLL { get; set; } = 0.0034; // Toe cliff
	[Range(11, 10000)]
	public double MaxCLL { get; set; } = 1300;
	[Range(0, 10)]
	public double BlackLuminance { get; set; } = 0.0;
	[Range(11, 10000)]
	public double WhiteLuminance { get; set; } = 265;
	[Range(0, 10)]
	public double SDRMinBrightness { get; set; } = 0.0;
	[Range(11, 10000)]
	public double SDRMaxBrightness { get; set; } = 100;
	[Range(-50, 50)]
	public double SDRBrightnessBoost { get; set; }

	public CustomDisplay SelectedDisplay { get; set; }
	public IEnumerable<CustomDisplay> Displays { get; set; }

	public string SelectedDisplayName => SelectedDisplay?.DisplayName;

	public ObservableCollection<string> ExistingProfiles { get; set; }
	public string SelectedExistingProfile { get; set; } = CreateANewProfile;
	public string NewProfileName { get; set; }
	public string ProfileName => SelectedExistingProfile == CreateANewProfile ? NewProfileName : SelectedExistingProfile;
	public bool IsHDR { get; set; } = true;
	public Dictionary<SaveOption, string> SaveOptions { get; } = Utils.EnumToDictionary<SaveOption>();
	public SaveOption SaveOption { get; set; } = SaveOption.InstallAndSetAsDefault;

	public bool SDRSettingsEnabled { get; set; }
	public Visibility VisibilityHDRSettings => IsHDR ? Visibility.Visible : Visibility.Collapsed;
	public bool IsLoadEnabled { get; set; }
	public bool SetMinMaxTml { get; set; } = true;
	public bool PrimariesEnabled { get; set; } = true;

	[Range(400, 10000)]
	public double ToneMappingFromLuminance { get; set; } = 1300;
	[Range(400, 10000)]
	public double ToneMappingToLuminance { get; set; } = 1300;


	public bool ToneMappingSettingsEnabled { get; set; }

	public bool BrightnessBoostSettingsEnabled { get; set; }

	[Range(.04, 25)]
	public double HdrBrightnessMultiplier { get; set; } = 1.0;

	[Range(.04, 25)]
	public double HdrGammaMultiplier { get; set; } = 0.1;

	[Range(0, 100)]
	public double WOLEDDesaturationCompensation { get; set; } = 95;

	public bool GenerateCubeLut { get; set; } = true;

	public override string this[string columnName]
	{
		get
		{
			var baseResult = base[columnName];

			if (columnName == nameof(SelectedDisplay))
			{
				UpdateDisplay();
			}
			else if (columnName == nameof(SDRTransferFunction))
			{
				ToneMappingSettingsEnabled = false;
				SDRSettingsEnabled = false;
				BrightnessBoostSettingsEnabled = true;
				OnPropertyChanged(nameof(ToneMappingSettingsEnabled));
				OnPropertyChanged(nameof(SDRSettingsEnabled));
				OnPropertyChanged(nameof(BrightnessBoostSettingsEnabled));
			}
			else if (columnName == nameof(SelectedExistingProfile))
			{
				IsLoadEnabled = SelectedExistingProfile != CreateANewProfile;
				OnPropertyChanged(nameof(IsLoadEnabled));
			}
			else if (columnName == nameof(PrimariesSource))
			{
				PrimariesEnabled = PrimariesSource == DisplayPrimariesSource.Custom;
				OnPropertyChanged(nameof(PrimariesEnabled));
				UpdateDisplay();
			}

			return baseResult;
		}
	}

	public override bool PreValidate()
	{
		return base.PreValidate() && Displays.Any();
	}

	public const string CreateANewProfile = "<create a new profile>";

	public ColorProfileViewModel(bool isHDR = true)
	{
		IsHDR = isHDR;

		RefreshDisplays();

		ExistingProfiles = new ObservableCollection<string> { CreateANewProfile };
		NewProfileName = $"CC_{(isHDR ? "HDR" : "SDR")}_profile_{DateTime.Now:yyyyMMddHHmmss}.icm";
		Description = $"CC_{(isHDR ? "HDR" : "SDR")}";

		if (Displays.Any())
		{
			SelectedDisplay = Displays.First();

			UpdateDisplay();
		}
	}

	public void RefreshDisplays()
	{
		Displays = Display.GetDisplays().Select(d => new CustomDisplay(d))
			.Where(d => IsHDR == d.IsHDRSupportedAndEnabled())
			.ToList();
	}

	public void Update()
	{
		foreach (var property in GetType().GetProperties().Where(p => !new[] { "Display", "PrimariesSource" }.Any(s => p.Name.Contains(s))))
		{
			OnPropertyChanged(property.Name);
		}
	}

	private void UpdateDisplay()
	{
		if (SelectedDisplay == null)
		{
			return;
		}

		var colorParams = CCD.GetColorParams(SelectedDisplay.DisplayName);

		if (colorParams.MaxLuminance > 0)
		{
			if (PrimariesSource == DisplayPrimariesSource.Windows)
			{
				var divider = colorParams.RedPointX <= 1 << 10 ? 1 << 10 : 1 << 20;

				RedPoint.X = (double)colorParams.RedPointX / divider;
				RedPoint.Y = (double)colorParams.RedPointY / divider;
				GreenPoint.X = (double)colorParams.GreenPointX / divider;
				GreenPoint.Y = (double)colorParams.GreenPointY / divider;
				BluePoint.X = (double)colorParams.BluePointX / divider;
				BluePoint.Y = (double)colorParams.BluePointY / divider;
				WhitePoint.X = (double)colorParams.WhitePointX / divider;
				WhitePoint.Y = (double)colorParams.WhitePointY / divider;
			}

			// Always use fixed defaults — the display-reported values vary
			// between launches and don't reflect the actual panel characteristics.
			WhiteLuminance = 1300;
			OnPropertyChanged(nameof(WhiteLuminance));
			BlackLuminance = 0.0;
			OnPropertyChanged(nameof(BlackLuminance));
			MinCLL = 0.00248;
			OnPropertyChanged(nameof(MinCLL));
			MaxCLL = 1300;
			OnPropertyChanged(nameof(MaxCLL));
		}

		if (PrimariesSource == DisplayPrimariesSource.EDID)
		{
			GetEDID();
		}

		if (!CCD.GetUsePerUserDisplayProfiles(SelectedDisplay.DisplayName))
		{
			if (MessageForms.QuestionYesNo("To be able to select/activate a color profile, user specific settings have to be enabled. Do you want to enable this?") == DialogResult.Yes)
			{
				CCD.SetUsePerUserDisplayProfiles(SelectedDisplay.DisplayName, true);
			}
		}

		ExistingProfiles = new ObservableCollection<string> { CreateANewProfile };

		var profiles = CCD.GetDisplayColorProfiles(SelectedDisplay.DisplayName);

		foreach (var profile in profiles)
		{
			ExistingProfiles.Add(profile);
		}
	}

	private void GetEDID()
	{
		if (SelectedDisplay == null)
		{
			return;
		}

		var path = SelectedDisplay.DevicePath;

		var edid = ColorProfileService.GetEDIDInternal(path);
		var primaries = edid?.DisplayParameters?.ChromaticityCoordinates;

		if (primaries == null)
		{
			return;
		}

		RedPoint.X = primaries.RedX;
		RedPoint.Y = primaries.RedY;
		GreenPoint.X = primaries.GreenX;
		GreenPoint.Y = primaries.GreenY;
		BluePoint.X = primaries.BlueX;
		BluePoint.Y = primaries.BlueY;
		WhitePoint.X = primaries.WhiteX;
		WhitePoint.Y = primaries.WhiteY;
	}

	internal RgbPrimaries GetDevicePrimaries()
	{
		return new RgbPrimaries(
			new CIExy { x = RedPoint.X, y = RedPoint.Y },
			new CIExy { x = GreenPoint.X, y = GreenPoint.Y },
			new CIExy { x = BluePoint.X, y = BluePoint.Y },
			new CIExy { x = WhitePoint.X, y = WhitePoint.Y });
	}
}
