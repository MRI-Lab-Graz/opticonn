import nibabel as nib
import numpy as np
import sys
import os


def check_atlas(path, name):
    print(f"Checking {name}...")
    if not os.path.exists(path):
        print(f"  Error: File not found at {path}")
        return None

    try:
        img = nib.load(path)
        print(f"  Shape: {img.shape}")
        print(f"  Affine:\n{img.affine}")
        print(f"  Voxel sizes: {img.header.get_zooms()}")
        return img
    except Exception as e:
        print(f"  Error loading NIfTI: {e}")
        return None


def main():
    dsi_studio_path = "/Applications/dsi_studio.app/Contents/MacOS/atlas/human"
    aal3_path = os.path.join(dsi_studio_path, "AAL3.nii.gz")
    # Compare with a standard DSI Studio atlas (usually MNI 1mm or 2mm)
    # BrainSeg is a standard label file.
    ref_path = os.path.join(dsi_studio_path, "BrainSeg.nii.gz")

    aal3_img = check_atlas(aal3_path, "AAL3 (New)")
    ref_img = check_atlas(ref_path, "BrainSeg (Reference)")

    if aal3_img and ref_img:
        print("\n--- Comparison ---")
        if aal3_img.shape == ref_img.shape:
            print("✅ Shapes match.")
        else:
            print(f"❌ Shape mismatch: AAL3 {aal3_img.shape} vs Ref {ref_img.shape}")

        if np.allclose(aal3_img.affine, ref_img.affine):
            print("✅ Affines match (Space is aligned).")
        else:
            print("❌ Affine mismatch (Space might differ).")
            print("Difference:\n", aal3_img.affine - ref_img.affine)


if __name__ == "__main__":
    try:
        import nibabel
    except ImportError:
        print("nibabel is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nibabel"])

    main()
