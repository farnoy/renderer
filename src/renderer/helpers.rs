pub(crate) mod command_util;

pub(crate) fn pick_lod<T>(lods: &[T], camera_pos: na::Point3<f32>, mesh_pos: na::Point3<f32>) -> &T {
    let distance_from_camera = (camera_pos - mesh_pos).magnitude();
    // TODO: fine-tune this later
    if distance_from_camera > 10.0 && lods.len() > 1 {
        lods.get(1).expect("empty index buffer LODs")
    } else {
        lods.first().expect("empty index buffer LODs")
    }
}
