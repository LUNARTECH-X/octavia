# PowerShell script to test subtitle translation endpoint

# Testing with query parameters - now WITH authentication
# First get a JWT token
$demoLoginUrl = "http://localhost:8000/api/auth/demo-login"
$token = ""

try {
    $loginResponse = Invoke-WebRequest -Uri $demoLoginUrl -Method POST -ContentType "application/json" -Body "{}"
    $loginData = $loginResponse.Content | ConvertFrom-Json
    $token = $loginData.token
    Write-Host "Got JWT token: $($token.Substring(0, 50))..."
} catch {
    Write-Host "Failed to get JWT token: $($_.Exception.Message)"
    exit 1
}

# Now test subtitle translation with authentication
$url = "http://localhost:8000/api/translate/subtitle-file?sourceLanguage=en&targetLanguage=es&format=srt"

# Create multipart form data (only for file)
$boundary = "----FormBoundary" + [Guid]::NewGuid().ToString("N")
$contentType = "multipart/form-data; boundary=$boundary"

# Read the file content
$fileContent = Get-Content -Path "test_subtitle.srt" -Raw -Encoding UTF8

# Build the multipart body (only file)
$bodyLines = @(
    "--$boundary",
    'Content-Disposition: form-data; name="file"; filename="test_subtitle.srt"',
    'Content-Type: text/plain',
    '',
    $fileContent,
    "--$boundary--"
)

$body = $bodyLines -join "`r`n"

# Make the request WITH authentication
try {
    $response = Invoke-WebRequest -Uri $url -Method POST -Headers @{
        "Authorization" = "Bearer $token"
        "Content-Type" = $contentType
    } -Body $body

    Write-Host "SUCCESS! Response Status:" $response.StatusCode
    Write-Host "Response Content:"
    Write-Host $response.Content
} catch {
    Write-Host "STILL ERROR:" $_.Exception.Message
    if ($_.Exception.Response) {
        Write-Host "Status Code:" $_.Exception.Response.StatusCode
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response Body:" $responseBody
    }
}
