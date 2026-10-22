# Install the sealpup command line on Windows.
#
#   irm https://raw.githubusercontent.com/sealresearch/gust/main/cli/install.ps1 | iex
#
# Settings, as environment variables:
#   SEALPUP_VERSION      a release tag such as v0.1.0 (default: the latest release)
#   SEALPUP_INSTALL      where to put the program (default: %LOCALAPPDATA%\Programs\sealpup)
#   SEALPUP_GITHUB_URL   download from a mirror instead of https://github.com
#   SEALPUP_NO_MODIFY_PATH   set it to leave your PATH alone

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repo = 'sealresearch/gust'
$base = if ($env:SEALPUP_GITHUB_URL) { $env:SEALPUP_GITHUB_URL.TrimEnd('/') } else { 'https://github.com' }
$dir = if ($env:SEALPUP_INSTALL) { $env:SEALPUP_INSTALL } else { Join-Path $env:LOCALAPPDATA 'Programs\sealpup' }
$release = if ($env:SEALPUP_VERSION) { "$base/$repo/releases/download/$($env:SEALPUP_VERSION)" } else { "$base/$repo/releases/latest/download" }
$asset = 'sealpup-windows-x64.exe'

$tmp = Join-Path ([IO.Path]::GetTempPath()) ('sealpup-' + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host ''
    Write-Host '  Installing sealpup (windows x64)'
    $web = New-Object Net.WebClient
    $web.DownloadFile("$release/SHA256SUMS", (Join-Path $tmp 'SHA256SUMS'))
    $web.DownloadFile("$release/$asset", (Join-Path $tmp $asset))

    $line = Get-Content (Join-Path $tmp 'SHA256SUMS') | Where-Object { $_ -match ('\*?' + [regex]::Escape($asset) + '$') } | Select-Object -First 1
    if (-not $line) { throw "the release has no checksum for $asset" }
    $expected = ($line -split '\s+')[0].ToLower()
    $actual = (Get-FileHash (Join-Path $tmp $asset) -Algorithm SHA256).Hash.ToLower()
    if ($expected -ne $actual) { throw 'the download does not match its checksum, so it was not installed' }

    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $target = Join-Path $dir 'sealpup.exe'
    Move-Item -Force (Join-Path $tmp $asset) $target

    $version = & $target --version
    Write-Host ''
    Write-Host "  $([char]0x2713) $version  $target"

    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $env:SEALPUP_NO_MODIFY_PATH -and ($user -split ';') -notcontains $dir) {
        [Environment]::SetEnvironmentVariable('Path', ($user.TrimEnd(';') + ';' + $dir), 'User')
        Write-Host ''
        Write-Host "  Added $dir to your PATH. Open a new terminal to use it."
    }
    Write-Host ''
    Write-Host '  Next: sealpup install relay-1'
    Write-Host ''
}
finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
