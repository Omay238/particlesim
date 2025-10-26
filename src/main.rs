use quarkstrom;

use egui;
use tokio;
use ultraviolet::DVec2 as Vec2;
use winit_input_helper::WinitInputHelper;

use once_cell::sync::Lazy;

// Used to communicate between the simulation and renderer threads
static PARTICLES: Lazy<egui::mutex::Mutex<Option<Vec<Particle>>>> =
    Lazy::new(|| egui::mutex::Mutex::new(None));
static RENDERER_CLONE: Lazy<egui::mutex::Mutex<Vec<Renderer>>> =
    Lazy::new(|| egui::mutex::Mutex::new(Vec::new()));
static STOPPED: Lazy<egui::mutex::Mutex<bool>> = Lazy::new(|| egui::mutex::Mutex::new(false));

fn get_tarkosky_lines() -> Vec<ultraviolet::Vec2> {
    vec![
        ultraviolet::Vec2::new(0.0, 0.0),
        ultraviolet::Vec2::new(1.0, 0.0),
        ultraviolet::Vec2::new(1.0, 1.0),
        ultraviolet::Vec2::new(2.0, 2.0),
        ultraviolet::Vec2::new(2.0, 1.0),
        ultraviolet::Vec2::new(3.0, 1.0),
        ultraviolet::Vec2::new(3.0, 0.0),
        ultraviolet::Vec2::new(4.0, 0.0),
        ultraviolet::Vec2::new(3.0, -1.0),
        ultraviolet::Vec2::new(2.0, -1.0),
        ultraviolet::Vec2::new(2.0, -2.0),
        ultraviolet::Vec2::new(1.0, -1.0),
        ultraviolet::Vec2::new(0.0, -1.0),
        ultraviolet::Vec2::new(0.0, -2.0),
        ultraviolet::Vec2::new(-1.0, -1.0),
        ultraviolet::Vec2::new(-2.0, -1.0),
        ultraviolet::Vec2::new(-2.0, -2.0),
        ultraviolet::Vec2::new(-3.0, -1.0),
        ultraviolet::Vec2::new(-3.0, 0.0),
        ultraviolet::Vec2::new(-4.0, 0.0),
        ultraviolet::Vec2::new(-3.0, 1.0),
        ultraviolet::Vec2::new(-2.0, 1.0),
        ultraviolet::Vec2::new(-2.0, 2.0),
        ultraviolet::Vec2::new(-1.0, 1.0),
        ultraviolet::Vec2::new(-1.0, 0.0),
        ultraviolet::Vec2::new(0.0, 0.0),
    ]
}

#[tokio::main]
async fn main() {
    let config = quarkstrom::Config {
        window_mode: quarkstrom::WindowMode::Windowed(1280, 720),
    };

    let tps_cap: Option<u32> = Some(60);

    let desired_frame_time =
        tps_cap.map(|tps| std::time::Duration::from_secs_f64(1.0 / tps as f64));

    let mut sim = Simulation::new(65536 * 2);

    tokio::spawn(async move {
        loop {
            let frame_timer = std::time::Instant::now();

            sim.update_simulation();
            // simulation.convert();

            // Cap tps
            if let Some(desired_frame_time) = desired_frame_time {
                while frame_timer.elapsed() < desired_frame_time {}
            }

            if *STOPPED.lock() {
                break;
            }
        }
    });

    quarkstrom::run::<Renderer>(config);
    *STOPPED.lock() = true;
}

#[derive(Clone)]
enum DemoType {
    Tarkosky,
}

#[derive(Clone)]
struct Renderer {
    pos: Vec2,
    scale: f32,
    demo_type: DemoType,
}

impl quarkstrom::Renderer for Renderer {
    fn new() -> Self {
        Self {
            pos: Vec2::new(0.0, 0.0),
            scale: 1.0,
            demo_type: DemoType::Tarkosky,
        }
    }

    fn input(&mut self, input: &WinitInputHelper, width: u16, height: u16) {
        if input.held_shift() {
            self.pos = Vec2::new(0.8, 0.0);
            self.scale = 0.1;
        } else {
            self.pos = Vec2::new(0.0, 0.0);
            self.scale = 1.0;
        }
    }

    fn render(&mut self, ctx: &mut quarkstrom::RenderContext) {
        ctx.set_view_pos(ultraviolet::Vec2::new(self.pos.x as f32, self.pos.y as f32));
        ctx.set_view_scale(self.scale);

        if let Some(particles) = PARTICLES.lock().clone() {
            ctx.clear_circles();
            ctx.clear_lines();

            for particle in particles {
                ctx.draw_circle(
                    ultraviolet::Vec2::new(particle.pos.x as f32, particle.pos.y as f32),
                    0.001,
                    [255, 255, 255, 255],
                );
            }
        }

        match self.demo_type {
            DemoType::Tarkosky => {
                let points = get_tarkosky_lines();

                for i in 0..points.len() - 1 {
                    ctx.draw_line(points[i] * 0.4, points[i + 1] * 0.4, [255, 255, 255, 255]);
                }

                ctx.draw_circle(ultraviolet::Vec2::new(0.8, 0.0), 0.002, [255, 0, 0, 128]);
            }
        }
    }

    fn gui(&mut self, ctx: &egui::Context) {
        return;
    }
}

// https://www.jeffreythompson.org/collision-detection/line-line.php
fn line_line(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> Option<Vec2> {
    let u_a = ((p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x))
        / ((p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y));
    let u_b = ((p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x))
        / ((p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y));

    if u_a >= 0.0 && u_a <= 1.0 && u_b >= 0.0 && u_b <= 1.0 {
        return Some(Vec2::new(
            p1.x + (u_a * (p2.x - p1.x)),
            p1.y + (u_a * (p2.y - p1.y)),
        ));
    }
    None
}

fn line_point(p1: Vec2, p2: Vec2, p3: Vec2) -> bool {
    // use squared distance because it's just comparisons
    let d1 = (p3.x - p1.x).powi(2) + (p3.y - p1.y).powi(2);
    let d2 = (p3.x - p2.x).powi(2) + (p3.y - p2.y).powi(2);

    let len = (p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2);

    if d1 + d2 >= len - 0.00001 && d1 + d2 <= len + 0.00001 {
        return true;
    }
    false
}

#[derive(Clone)]
struct Particle {
    pos: Vec2,
    angle: f64,
}

struct Simulation {
    particles: Vec<Particle>,
}

impl Simulation {
    fn new(num_particles: usize) -> Self {
        let mut particles: Vec<Particle> = Vec::new();

        for i in 0..num_particles {
            particles.push(Particle {
                pos: Vec2::new(-0.8, 0.0),
                angle: std::f64::consts::TAU * (i as f64 / num_particles as f64),
            });
        }

        Self { particles }
    }

    fn update_simulation(&mut self) {
        for particle in &mut self.particles {
            let mut goal_pos =
                particle.pos + Vec2::new(particle.angle.cos(), particle.angle.sin()) * 0.001;

            let points = get_tarkosky_lines();

            for i in 0..points.len() - 1 {
                if let Some(intersection_point) = line_line(
                    particle.pos,
                    goal_pos,
                    Vec2::new(points[i].x as f64, points[i].y as f64) * 0.4,
                    Vec2::new(points[i + 1].x as f64, points[i + 1].y as f64) * 0.4,
                ) {
                    particle.pos = intersection_point;

                    // lmaoooooo i used desmos to find this xd i don't know how it works
                    // https://www.desmos.com/calculator/popf3ryijo

                    // i couldn't find out how to get the normal without just arctan
                    let normal = ((points[i].x as f64 - points[i + 1].x as f64)
                        / (points[i].y as f64 - points[i + 1].y as f64))
                        .atan()
                        + std::f64::consts::FRAC_PI_2;
                    let mut normal_vec = Vec2::new(normal.sin(), normal.cos());

                    let dir_vec = (goal_pos - particle.pos).normalized();

                    if dir_vec.dot(normal_vec) > 0.0 {
                        normal_vec = -normal_vec;
                    }

                    let reflected = dir_vec - 2.0 * dir_vec.dot(normal_vec) * normal_vec;

                    particle.angle = f64::atan2(reflected.y, reflected.x);

                    goal_pos = intersection_point
                        + Vec2::new(particle.angle.cos(), particle.angle.sin()) * 0.001;
                } else if line_point(
                    particle.pos,
                    goal_pos,
                    Vec2::new(points[i].x as f64, points[i].y as f64) * 0.4,
                ) {
                    particle.angle -= std::f64::consts::PI;
                    goal_pos = Vec2::new(points[i].x as f64, points[i].y as f64) * 0.4
                        + Vec2::new(particle.angle.cos(), particle.angle.sin()) * 0.001;
                }
            }

            particle.pos = goal_pos;
        }

        *PARTICLES.lock() = Some(self.particles.clone());
    }
}
