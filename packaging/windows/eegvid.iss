; Inno Setup script: Windows installer for EEG Video Recorder.
; Build the app first (pyinstaller packaging/eegvid.spec --noconfirm), then:
;   iscc /DAppVersion=0.1.0 packaging\windows\eegvid.iss
; Output: dist\EEGVideoRecorder-<version>-windows-setup.exe
;
; Installs per user by default (no admin rights needed); the dialog offers
; "all users" too. The BioSemi USB (WinUSB) driver is NOT installed here: it
; comes with BioSemi's ActiView download, and any PC that runs ActiView has it.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6B0E8F52-3C1A-4C4F-9B7E-EE6E0A1D2C11}
AppName=EEG Video Recorder
AppVersion={#AppVersion}
AppPublisher=EEG lab
DefaultDirName={autopf}\EEG Video Recorder
DefaultGroupName=EEG Video Recorder
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=EEGVideoRecorder-{#AppVersion}-windows-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\EEGVideoRecorder.exe
InfoBeforeFile=..\INSTALL.txt

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\..\dist\EEGVideoRecorder\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\INSTALL.txt"; DestDir: "{app}"; DestName: "Read Me First.txt"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\EEG Video Recorder"; Filename: "{app}\EEGVideoRecorder.exe"
Name: "{autodesktop}\EEG Video Recorder"; Filename: "{app}\EEGVideoRecorder.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\EEGVideoRecorder.exe"; Description: "Start EEG Video Recorder"; Flags: nowait postinstall skipifsilent
