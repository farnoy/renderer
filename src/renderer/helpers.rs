#[cfg(not(feature = "no_profiling"))]
pub(crate) const MP_INDIAN_RED: u32 = 0xcd5c5c;

pub(crate) fn pick_lod<T>(lods: &[T], camera_pos: na::Point3<f32>, mesh_pos: na::Point3<f32>) -> &T {
    let distance_from_camera = (camera_pos - mesh_pos).magnitude();
    // TODO: fine-tune this later
    if distance_from_camera > 10.0 {
        lods.last().expect("empty index buffer LODs")
    } else {
        lods.first().expect("empty index buffer LODs")
    }
}
