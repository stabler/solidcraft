//! Whether a body is a solid of revolution about a given axis.

use solidcore_record::Vec3;
use solidcore_record::brep::{Brep, Loop};
use solidcraft_geom::Vec3Geom;

use crate::body::Body;
use crate::surface_kind::{FaceSurface, surface_of};

/// Whether every face of `body` is unchanged by a turn about the line through `origin` along
/// `axis`: a plane square to that line, or a cylinder, cone, sphere or torus on it. A mesh, a
/// freeform face, or a face the record does not classify, is not.
pub fn turned_about(body: &Body, origin: Vec3, axis: Vec3) -> bool {
    let Some(axis) = axis.normalized() else {
        return false;
    };
    if !origin.x.is_finite() || !origin.y.is_finite() || !origin.z.is_finite() || body.is_mesh() {
        return false;
    }
    let record = body.record();
    if record.faces.is_empty() {
        return false;
    }
    let tol = record.tolerance.max(1e-9);
    (0..record.faces.len()).all(|face| match surface_of(record, face) {
        Some(kind) => face_turned(kind, origin, axis, tol),
        None => false,
    })
}

/// A closed solid of revolution about the line through `origin` along `axis`. An open sheet of
/// revolution surfaces, and a solid with no material, is not: turning one would copy a patch.
pub fn closed_turn(body: &Body, origin: Vec3, axis: Vec3) -> bool {
    if !turned_about(body, origin, axis) || body.is_surface() || body.record().material_regions().is_empty() {
        return false;
    }
    match crate::measure_mass(body) {
        Ok(m) => m.volume.is_finite() && m.volume > 1e-6,
        Err(_) => false,
    }
}

/// The change from `before` to `after` lies in the box from `min` to `max`. A vertex of a face
/// that is not a surface of revolution about the axis lies in the box. Every other vertex of
/// `after` lies in the box or on a vertex of `before`.
pub fn local_change(after: &Body, before: &Body, origin: Vec3, axis: Vec3, min: Vec3, max: Vec3) -> bool {
    let Some(axis) = axis.normalized() else {
        return false;
    };
    if after.is_mesh() || before.is_mesh() || ![origin.x, origin.y, origin.z, min.x, min.y, min.z, max.x, max.y, max.z].iter().all(|c| c.is_finite())
    {
        return false;
    }
    let added = after.record();
    let prior = before.record();
    if added.faces.is_empty() || added.vertices.is_empty() {
        return false;
    }
    let tol = added.tolerance.max(prior.tolerance).max(1e-9);
    let earlier: Vec<Vec3> = prior.vertices.iter().map(|v| Vec3::from(v.point)).collect();
    for vertex in &added.vertices {
        let p = Vec3::from(vertex.point);
        if in_box(p, min, max, tol) || earlier.iter().any(|q| (*q - p).len() <= tol) {
            continue;
        }
        return false;
    }
    (0..added.faces.len()).all(|face| {
        let turned = surface_of(added, face).is_some_and(|kind| face_turned(kind, origin, axis, tol));
        turned || face_points(added, face).is_some_and(|points| points.iter().all(|p| in_box(*p, min, max, tol)))
    })
}

fn in_box(p: Vec3, min: Vec3, max: Vec3, tol: f64) -> bool {
    p.x >= min.x - tol && p.x <= max.x + tol && p.y >= min.y - tol && p.y <= max.y + tol && p.z >= min.z - tol && p.z <= max.z + tol
}

/// The vertices on `face`'s loops. `None` when a loop names an edge or a vertex the record lacks.
fn face_points(record: &Brep, face: usize) -> Option<Vec<Vec3>> {
    let face = record.faces.get(face)?;
    let mut points = Vec::new();
    for lp in &face.loops {
        match lp {
            Loop::Edges { edges } => {
                for edge in edges {
                    let edge = record.edges.get(edge.edge as usize)?;
                    for vertex in edge.vertices {
                        points.push(Vec3::from(record.vertices.get(vertex as usize)?.point));
                    }
                }
            }
            Loop::Vertex { vertex } => points.push(Vec3::from(record.vertices.get(*vertex as usize)?.point)),
        }
    }
    Some(points)
}

/// `point` lies on the line through `origin` along the unit `axis`, within `tol` mm.
fn on_axis(origin: Vec3, axis: Vec3, point: Vec3, tol: f64) -> bool {
    let w = point - origin;
    (w - axis * w.dot(axis)).len() <= tol
}

/// The unit directions are the same line, either way.
fn parallel(a: Vec3, b: Vec3) -> bool {
    let (Some(a), Some(b)) = (a.normalized(), b.normalized()) else {
        return false;
    };
    a.cross(b).len() <= 1e-8
}

fn face_turned(kind: FaceSurface, origin: Vec3, axis: Vec3, tol: f64) -> bool {
    match kind {
        FaceSurface::Plane { normal, .. } => parallel(Vec3::from(normal), axis),
        FaceSurface::Cylinder { origin: o, axis: a, .. } | FaceSurface::Torus { center: o, axis: a, .. } => {
            parallel(Vec3::from(a), axis) && on_axis(origin, axis, o, tol)
        }
        FaceSurface::Cone { apex, axis: a, .. } => parallel(Vec3::from(a), axis) && on_axis(origin, axis, apex, tol),
        FaceSurface::Sphere { center, .. } => on_axis(origin, axis, center, tol),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{box_solid, cylinder};

    #[test]
    fn a_cylinder_turns_about_its_axis() {
        let body = cylinder(Vec3::ZERO, Vec3::Z, 5.0, 10.0).unwrap();
        assert!(turned_about(&body, Vec3::ZERO, Vec3::Z));
        assert!(!turned_about(&body, Vec3::ZERO, Vec3::X));
    }

    #[test]
    fn a_box_does_not_turn_about_an_axis() {
        let body = box_solid(Vec3::ZERO, Vec3::new(10.0, 20.0, 30.0)).unwrap();
        assert!(!turned_about(&body, Vec3::new(5.0, 10.0, 15.0), Vec3::Z));
    }

    #[test]
    fn a_closed_cylinder_is_a_turn_and_a_box_is_not_a_local_change() {
        let cylinder = cylinder(Vec3::ZERO, Vec3::Z, 5.0, 10.0).unwrap();
        assert!(closed_turn(&cylinder, Vec3::ZERO, Vec3::Z));
        assert!(!closed_turn(&cylinder, Vec3::ZERO, Vec3::X));
        let block = box_solid(Vec3::new(20.0, 20.0, 0.0), Vec3::new(30.0, 30.0, 10.0)).unwrap();
        let window = (Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        assert!(!local_change(&block, &cylinder, Vec3::ZERO, Vec3::Z, window.0, window.1));
    }
}
