param([Parameter(Mandatory=$true)][string]$Spec)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing
$plots=Get-Content -LiteralPath $Spec -Raw -Encoding UTF8 | ConvertFrom-Json
foreach($plot in $plots) {
    $chart=New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    try {
        $chart.Width=1500; $chart.Height=900; $chart.BackColor=[System.Drawing.Color]::White
        $chart.Titles.Add([string]$plot.title) | Out-Null
        $chart.Titles[0].Font=New-Object System.Drawing.Font('Segoe UI',18)
        $area=New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea('main')
        $area.AxisX.Title='Square matrix dimension (M=N=K)'
        $area.AxisY.Title=[string]$plot.ylabel
        $area.AxisX.IsLogarithmic=$true; $area.AxisX.LogarithmBase=2
        $area.AxisX.Minimum=32; $area.AxisX.Maximum=1024
        $area.AxisY.IsLogarithmic=[bool]$plot.logy
        $area.AxisX.LabelStyle.Format='0'; $area.AxisY.LabelStyle.Format='0.###'
        $area.AxisX.MajorGrid.LineColor=[System.Drawing.Color]::Gainsboro
        $area.AxisY.MajorGrid.LineColor=[System.Drawing.Color]::Gainsboro
        $area.AxisX.TitleFont=New-Object System.Drawing.Font('Segoe UI',12)
        $area.AxisY.TitleFont=New-Object System.Drawing.Font('Segoe UI',12)
        $chart.ChartAreas.Add($area)
        $legend=New-Object System.Windows.Forms.DataVisualization.Charting.Legend('Legend')
        $legend.Docking='Bottom';$legend.Font=New-Object System.Drawing.Font('Segoe UI',11)
        $chart.Legends.Add($legend)
        foreach($line in $plot.series) {
            $series=New-Object System.Windows.Forms.DataVisualization.Charting.Series([string]$line.name)
            $series.ChartType='Line';$series.BorderWidth=3;$series.MarkerStyle='Circle';$series.MarkerSize=7
            foreach($point in $line.points) {$series.Points.AddXY([double]$point[0],[double]$point[1]) | Out-Null}
            $chart.Series.Add($series)
        }
        $chart.SaveImage([string]$plot.output,'Png')
    } finally {$chart.Dispose()}
}
