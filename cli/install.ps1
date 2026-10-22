# Install the seal command line on Windows.
#
#   irm https://raw.githubusercontent.com/sealresearch/gust/main/cli/install.ps1 | iex
#
# Settings, as environment variables:
#   SEAL_VERSION      a release tag such as v0.1.0 (default: the latest release)
#   SEAL_INSTALL      where to put the program (default: %LOCALAPPDATA%\Programs\seal)
#   SEAL_GITHUB_URL   download from a mirror instead of https://github.com
#   SEAL_NO_MODIFY_PATH   set it to leave your PATH alone

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repo = 'sealresearch/gust'
$base = if ($env:SEAL_GITHUB_URL) { $env:SEAL_GITHUB_URL.TrimEnd('/') } else { 'https://github.com' }
$dir = if ($env:SEAL_INSTALL) { $env:SEAL_INSTALL } else { Join-Path $env:LOCALAPPDATA 'Programs\seal' }
$release = if ($env:SEAL_VERSION) { "$base/$repo/releases/download/$($env:SEAL_VERSION)" } else { "$base/$repo/releases/latest/download" }
$asset = 'seal-windows-x64.exe'

$tmp = Join-Path ([IO.Path]::GetTempPath()) ('seal-' + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host ''
    Write-Host '  Installing seal (windows x64)'
    $web = New-Object Net.WebClient
    $web.DownloadFile("$release/SHA256SUMS", (Join-Path $tmp 'SHA256SUMS'))
    $web.DownloadFile("$release/$asset", (Join-Path $tmp $asset))

    $line = Get-Content (Join-Path $tmp 'SHA256SUMS') | Where-Object { $_ -match ('\*?' + [regex]::Escape($asset) + '$') } | Select-Object -First 1
    if (-not $line) { throw "the release has no checksum for $asset" }
    $expected = ($line -split '\s+')[0].ToLower()
    $actual = (Get-FileHash (Join-Path $tmp $asset) -Algorithm SHA256).Hash.ToLower()
    if ($expected -ne $actual) { throw 'the download does not match its checksum, so it was not installed' }

    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $target = Join-Path $dir 'seal.exe'
    Move-Item -Force (Join-Path $tmp $asset) $target

    $version = & $target --version
    Write-Host ''
    Write-Host "  $([char]0x2713) $version  $target"

    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $env:SEAL_NO_MODIFY_PATH -and ($user -split ';') -notcontains $dir) {
        [Environment]::SetEnvironmentVariable('Path', ($user.TrimEnd(';') + ';' + $dir), 'User')
        Write-Host ''
        Write-Host "  Added $dir to your PATH. Open a new terminal to use it."
    }
    Write-Host ''
    Write-Host '  Next: seal install relay-1'
    Write-Host ''
}
finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
