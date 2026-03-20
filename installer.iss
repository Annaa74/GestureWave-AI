[Setup]
AppName=GestureWave AI
AppVersion=2.1
DefaultDirName={autopf}\GestureWaveAI
DefaultGroupName=GestureWave AI
UninstallDisplayIcon={app}\GestureWaveAI.exe
Compression=lzma2
SolidCompression=yes
OutputDir=dist
OutputBaseFilename=GestureWaveAI_Installer

[Files]
Source: "dist\GestureWaveAI.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\GestureWave AI"; Filename: "{app}\GestureWaveAI.exe"
Name: "{autodesktop}\GestureWave AI"; Filename: "{app}\GestureWaveAI.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"
