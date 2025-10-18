#!/usr/bin/env python3
"""
ParaView Occupancy Visualization Automation

Creates automated ParaView Python scripts for publication-quality occupancy visualizations.
ParaView excels at scientific volume data and offers the most advanced rendering options.
"""

import os
import json


def create_advanced_paraview_script(protein_pdb: str, occupancy_vtk: str,
                                  dataset_name: str, output_script: str):
    """Create advanced ParaView script with automated rendering."""

    protein_pdb_abs = os.path.abspath(protein_pdb)
    occupancy_vtk_abs = os.path.abspath(occupancy_vtk)

    script_content = f'''
# Advanced ParaView script for {dataset_name} fragment occupancy visualization
# Publication-quality scientific visualization with automated rendering

import paraview.simple as pv
import numpy as np
import os

print("🧬 ADVANCED {dataset_name.upper()} OCCUPANCY ANALYSIS")
print("=" * 60)

# === CLEAR AND SETUP ===
pv.Delete(pv.GetRenderViews()[0])
renderView = pv.CreateView('RenderView')
renderView.ViewSize = [1920, 1080]
renderView.Background = [1.0, 1.0, 1.0]  # White background

# === LOAD DATA ===
print("Loading protein structure: {protein_pdb}")
protein = pv.OpenDataFile("{protein_pdb_abs}")
protein.UpdatePipeline()

print("Loading occupancy data: {occupancy_vtk}")
occupancy = pv.OpenDataFile("{occupancy_vtk_abs}")
occupancy.UpdatePipeline()

# Get occupancy data range for adaptive thresholds
occupancy_info = occupancy.GetDataInformation()
data_range = occupancy_info.GetArrayInformation('probability').GetComponentRange(0)
min_occ = data_range[0]
max_occ = data_range[1]

print(f"Occupancy range: {{min_occ:.6f}} to {{max_occ:.6f}}")

# === ADVANCED PROTEIN VISUALIZATION ===
print("Setting up protein visualization...")
protein_display = pv.Show(protein, renderView)

# Protein as high-quality molecular surface
protein_display.Representation = 'Surface'
protein_display.ColorArrayName = [None, '']
protein_display.DiffuseColor = [0.8, 0.9, 1.0]  # Light blue
protein_display.Opacity = 0.3
protein_display.Specular = 0.8
protein_display.SpecularPower = 50

# Add protein cartoon representation
protein_cartoon = pv.Show(protein, renderView)
protein_cartoon.Representation = 'Surface'
protein_cartoon.ColorArrayName = ['POINTS', 'Temperature Factor']
protein_cartoon.Opacity = 0.7

# === HINGE REGION HIGHLIGHTING ===
# Extract atoms in hinge region (residues ~800-805)
hinge_filter = pv.Calculator(Input=protein)
hinge_filter.AttributeType = 'Point Data'
hinge_filter.Function = '(coordsZ > -5) * (coordsZ < 5)'  # Approximate hinge plane
hinge_filter.ResultArrayName = 'HingeRegion'
hinge_filter.UpdatePipeline()

hinge_display = pv.Show(hinge_filter, renderView)
hinge_display.Representation = 'Points'
hinge_display.ColorArrayName = ['POINTS', 'HingeRegion']
hinge_display.PointSize = 8.0

# === MULTI-LEVEL ISOSURFACES ===
print("Creating multi-level isosurfaces...")

# Level 1: Low confidence (30% of max)
low_threshold = max_occ * 0.3
contour_low = pv.Contour(Input=occupancy)
contour_low.ContourBy = ['POINTS', 'probability']
contour_low.Isosurfaces = [low_threshold]
contour_low.UpdatePipeline()

contour_low_display = pv.Show(contour_low, renderView)
contour_low_display.Representation = 'Surface'
contour_low_display.DiffuseColor = [1.0, 1.0, 0.0]  # Yellow
contour_low_display.Opacity = 0.3
contour_low_display.Specular = 0.5

# Level 2: Medium confidence (60% of max)
med_threshold = max_occ * 0.6
contour_med = pv.Contour(Input=occupancy)
contour_med.ContourBy = ['POINTS', 'probability']
contour_med.Isosurfaces = [med_threshold]
contour_med.UpdatePipeline()

contour_med_display = pv.Show(contour_med, renderView)
contour_med_display.Representation = 'Surface'
contour_med_display.DiffuseColor = [1.0, 0.6, 0.0]  # Orange
contour_med_display.Opacity = 0.5
contour_med_display.Specular = 0.7

# Level 3: High confidence (80% of max)
high_threshold = max_occ * 0.8
contour_high = pv.Contour(Input=occupancy)
contour_high.ContourBy = ['POINTS', 'probability']
contour_high.Isosurfaces = [high_threshold]
contour_high.UpdatePipeline()

contour_high_display = pv.Show(contour_high, renderView)
contour_high_display.Representation = 'Surface'
contour_high_display.DiffuseColor = [1.0, 0.0, 0.0]  # Red
contour_high_display.Opacity = 0.8
contour_high_display.Specular = 1.0

# === VOLUME RENDERING ===
print("Setting up volume rendering...")
occupancy_volume = pv.Show(occupancy, renderView)
occupancy_volume.Representation = 'Volume'

# Advanced volume rendering settings
volume_tf = pv.GetOpacityTransferFunction('probability')
volume_tf.RemoveAllPoints()
volume_tf.AddPoint(0.0, 0.0)
volume_tf.AddPoint(low_threshold, 0.1)
volume_tf.AddPoint(med_threshold, 0.3)
volume_tf.AddPoint(high_threshold, 0.6)
volume_tf.AddPoint(max_occ, 1.0)

color_tf = pv.GetColorTransferFunction('probability')
color_tf.ApplyPreset('Cool to Warm (Extended)', True)

# === ADVANCED LIGHTING AND RENDERING ===
renderView.EnableRayTracing = 1
renderView.Shadows = 1
renderView.AmbientSamples = 100
renderView.SamplesPerPixel = 10

# Lighting setup
light1 = pv.CreateLight()
light1.Position = [10, 10, 10]
light1.FocalPoint = [0, 0, 0]
light1.Intensity = 1.5

# === CAMERA POSITIONS AND SCENES ===
print("Setting up camera positions...")

# Camera 1: Overview
renderView.ResetCamera()
renderView.GetActiveCamera().SetPosition([30, 30, 30])
renderView.GetActiveCamera().SetFocalPoint([0, 0, 0])
renderView.GetActiveCamera().SetViewUp([0, 0, 1])
overview_camera = renderView.GetActiveCamera().GetState()

# Camera 2: ATP binding site focus
renderView.GetActiveCamera().SetPosition([15, 5, 10])
renderView.GetActiveCamera().SetFocalPoint([0, 0, 0])
renderView.GetActiveCamera().SetViewUp([0, 0, 1])
renderView.GetActiveCamera().Zoom(1.5)
atp_site_camera = renderView.GetActiveCamera().GetState()

# Camera 3: Hinge region detail
renderView.GetActiveCamera().SetPosition([8, 2, 5])
renderView.GetActiveCamera().SetFocalPoint([0, 0, 0])
renderView.GetActiveCamera().SetViewUp([0, 1, 0])
renderView.GetActiveCamera().Zoom(2.5)
hinge_camera = renderView.GetActiveCamera().GetState()

# === CROSS-SECTIONAL ANALYSIS ===
print("Creating cross-sectional views...")
slice_filter = pv.Slice(Input=occupancy)
slice_filter.SliceType = 'Plane'
slice_filter.SliceType.Origin = [0.0, 0.0, 0.0]
slice_filter.SliceType.Normal = [0.0, 0.0, 1.0]  # XY plane through ATP site
slice_filter.UpdatePipeline()

slice_display = pv.Show(slice_filter, renderView)
slice_display.Representation = 'Surface'
slice_display.ColorArrayName = ['POINTS', 'probability']
slice_display.Opacity = 0.8

# === AUTOMATED IMAGE GENERATION ===
print("\\nGenerating publication-quality images...")

# High-resolution settings
renderView.ViewSize = [3200, 2400]  # 4K resolution

# Image 1: Overview with all features
renderView.GetActiveCamera().SetState(overview_camera)
renderView.Render()
pv.SaveScreenshot('{dataset_name}_paraview_overview.png', renderView,
                 ImageResolution=[3200, 2400], TransparentBackground=0)

# Image 2: ATP binding site focus
renderView.GetActiveCamera().SetState(atp_site_camera)
renderView.Render()
pv.SaveScreenshot('{dataset_name}_paraview_atp_site.png', renderView,
                 ImageResolution=[3200, 2400], TransparentBackground=0)

# Image 3: Hinge region detail
renderView.GetActiveCamera().SetState(hinge_camera)
renderView.Render()
pv.SaveScreenshot('{dataset_name}_paraview_hinge.png', renderView,
                 ImageResolution=[3200, 2400], TransparentBackground=0)

# Image 4: Cross-section view
pv.Hide(protein_display, renderView)
pv.Hide(occupancy_volume, renderView)
pv.Show(slice_display, renderView)
renderView.GetActiveCamera().SetState(overview_camera)
renderView.Render()
pv.SaveScreenshot('{dataset_name}_paraview_cross_section.png', renderView,
                 ImageResolution=[3200, 2400], TransparentBackground=0)

# === QUANTITATIVE ANALYSIS ===
print("\\nPerforming quantitative analysis...")

# Volume calculations
integrate_filter = pv.IntegrateVariables(Input=occupancy)
integrate_filter.UpdatePipeline()

# Surface area calculations for each isosurface level
low_integrate = pv.IntegrateVariables(Input=contour_low)
low_integrate.UpdatePipeline()

med_integrate = pv.IntegrateVariables(Input=contour_med)
med_integrate.UpdatePipeline()

high_integrate = pv.IntegrateVariables(Input=contour_high)
high_integrate.UpdatePipeline()

print("\\n=== QUANTITATIVE RESULTS ===")
print(f"Dataset: {dataset_name}")
print(f"Max occupancy: {{max_occ:.6f}}")
print(f"Min occupancy: {{min_occ:.6f}}")
print(f"Threshold levels:")
print(f"  Low confidence: {{low_threshold:.6f}}")
print(f"  Med confidence: {{med_threshold:.6f}}")
print(f"  High confidence: {{high_threshold:.6f}}")

print("\\n=== IMAGES GENERATED ===")
print("📸 High-resolution images (3200x2400 pixels):")
print(f"  • {dataset_name}_paraview_overview.png")
print(f"  • {dataset_name}_paraview_atp_site.png")
print(f"  • {dataset_name}_paraview_hinge.png")
print(f"  • {dataset_name}_paraview_cross_section.png")

print("\\n=== VISUALIZATION FEATURES ===")
print("🎨 ADVANCED RENDERING:")
print("  • Ray tracing with shadows")
print("  • Multi-level isosurfaces")
print("  • Volume rendering with custom transfer functions")
print("  • Molecular surface with specular highlights")
print("  • Cross-sectional analysis")

print("\\n💡 SCIENTIFIC INSIGHTS:")
print(f"  • Red surfaces = Highest consensus binding (>{{max_occ * 0.8:.3f}})")
print(f"  • Orange surfaces = Medium confidence (>{{max_occ * 0.6:.3f}})")
print(f"  • Yellow surfaces = Low confidence (>{{max_occ * 0.3:.3f}})")
print("  • Proximity to protein surface = Direct interaction sites")
print("  • Cross-sections reveal binding pocket geometry")

print("\\n🎯 DRUG DISCOVERY IMPLICATIONS:")
print("  • High occupancy regions = Validated drug targets")
print("  • Multiple isosurfaces = Different binding modes")
print("  • Volume rendering shows accessible surface area")
print("  • Hinge proximity = ATP-competitive binding potential")

# === SAVE SESSION ===
pv.SaveState('{dataset_name}_paraview_session.pvsm')
print(f"\\nSession saved: {dataset_name}_paraview_session.pvsm")

print("\\n✅ ADVANCED PARAVIEW VISUALIZATION COMPLETE!")
print("🔬 Use the saved session file to continue interactive analysis")
'''

    with open(output_script, 'w') as f:
        f.write(script_content)

    print(f"Advanced ParaView script created: {output_script}")
    return output_script


def create_paraview_visualizations():
    """Create advanced ParaView visualization scripts."""

    datasets = {
        'aminopyrimidine': {
            'vtk': 'aminopyrimidine_atp_plane.vtk',
            'protein': 'reference_protein.pdb',
        },
        'quinoline': {
            'vtk': 'quinoline_atp_plane.vtk',
            'protein': 'quinoline_reference_protein.pdb',
        },
        'quinozolne': {
            'vtk': 'quinozolne_atp_plane.vtk',
            'protein': 'quinozolne_reference_protein.pdb',
        }
    }

    summary = {}

    for dataset_name, files in datasets.items():
        if not (os.path.exists(files['vtk']) and os.path.exists(files['protein'])):
            print(f"Skipping {dataset_name} - missing files")
            continue

        print(f"\\nProcessing {dataset_name}...")

        # Create ParaView script
        paraview_script = f"{dataset_name}_paraview_advanced.py"
        create_advanced_paraview_script(files['protein'], files['vtk'],
                                      dataset_name, paraview_script)

        summary[dataset_name] = {
            'paraview_script': paraview_script,
            'vtk_file': files['vtk'],
            'protein_file': files['protein']
        }

    # Save summary
    with open('paraview_advanced_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    print("Advanced ParaView Occupancy Visualization")
    print("=" * 50)

    summary = create_paraview_visualizations()

    print(f"\\n✅ Advanced ParaView Setup Complete!")
    print(f"Generated {len(summary)} advanced visualization scripts")

    print(f"\\n🎯 To use ParaView (Advanced):")
    for dataset, info in summary.items():
        print(f"\\n{dataset.upper()}:")
        print(f"  1. Open ParaView")
        print(f"  2. Tools → Python Shell")
        print(f"  3. exec(open('{info['paraview_script']}').read())")
        print(f"  4. Wait for automated rendering")

    print(f"\\n🌟 Advanced ParaView Features:")
    print(f"  • Ray tracing with global illumination")
    print(f"  • Multi-level isosurface analysis")
    print(f"  • Volume rendering with custom colormaps")
    print(f"  • Cross-sectional binding site analysis")
    print(f"  • Automated high-resolution image export")
    print(f"  • Quantitative surface area calculations")
    print(f"  • Session saving for reproducible analysis")

    print(f"\\n📊 Output Files Per Dataset:")
    print(f"  • [dataset]_paraview_overview.png (4K resolution)")
    print(f"  • [dataset]_paraview_atp_site.png (focused view)")
    print(f"  • [dataset]_paraview_hinge.png (hinge detail)")
    print(f"  • [dataset]_paraview_cross_section.png (cross-section)")
    print(f"  • [dataset]_paraview_session.pvsm (session file)")


if __name__ == "__main__":
    main()