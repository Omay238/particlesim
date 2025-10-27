use quarkstrom;

use egui;
use tokio;
use ultraviolet::DVec2 as Vec2;
use winit_input_helper::WinitInputHelper;

use once_cell::sync::Lazy;

// Used to communicate between the simulation and renderer threads
static PARTICLES: Lazy<egui::mutex::Mutex<Option<Vec<Vec<Particle>>>>> =
    Lazy::new(|| egui::mutex::Mutex::new(vec![vec![]].into()));
static STOPPED: Lazy<egui::mutex::Mutex<bool>> = Lazy::new(|| egui::mutex::Mutex::new(false));
static DEMO_TYPE: Lazy<egui::mutex::Mutex<DemoType>> = Lazy::new(|| egui::mutex::Mutex::new(DemoType::Castro));

// this function lowkey sucks really hard lmao
fn get_lines(demo_type: &DemoType) -> Vec<(ultraviolet::Vec2, ultraviolet::Vec2, ultraviolet::Vec2)> {
    match demo_type {
        DemoType::Tarkosky => vec![
            (
                ultraviolet::Vec2::new(0.0, 0.0),
                ultraviolet::Vec2::new(1.0, 0.0),
                ultraviolet::Vec2::new(1.0, 1.0),
            ),
            (
                ultraviolet::Vec2::new(1.0, 0.0),
                ultraviolet::Vec2::new(1.0, 1.0),
                ultraviolet::Vec2::new(2.0, 2.0),
            ),
            (
                ultraviolet::Vec2::new(1.0, 1.0),
                ultraviolet::Vec2::new(2.0, 2.0),
                ultraviolet::Vec2::new(2.0, 1.0),
            ),
            (
                ultraviolet::Vec2::new(2.0, 2.0),
                ultraviolet::Vec2::new(2.0, 1.0),
                ultraviolet::Vec2::new(3.0, 1.0),
            ),
            (
                ultraviolet::Vec2::new(2.0, 1.0),
                ultraviolet::Vec2::new(3.0, 1.0),
                ultraviolet::Vec2::new(3.0, 0.0),
            ),
            (
                ultraviolet::Vec2::new(3.0, 1.0),
                ultraviolet::Vec2::new(3.0, 0.0),
                ultraviolet::Vec2::new(4.0, 0.0),
            ),
            (
                ultraviolet::Vec2::new(3.0, 0.0),
                ultraviolet::Vec2::new(4.0, 0.0),
                ultraviolet::Vec2::new(3.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(4.0, 0.0),
                ultraviolet::Vec2::new(3.0, -1.0),
                ultraviolet::Vec2::new(2.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(3.0, -1.0),
                ultraviolet::Vec2::new(2.0, -1.0),
                ultraviolet::Vec2::new(2.0, -2.0),
            ),
            (
                ultraviolet::Vec2::new(2.0, -1.0),
                ultraviolet::Vec2::new(2.0, -2.0),
                ultraviolet::Vec2::new(1.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(2.0, -2.0),
                ultraviolet::Vec2::new(1.0, -1.0),
                ultraviolet::Vec2::new(0.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(1.0, -1.0),
                ultraviolet::Vec2::new(0.0, -1.0),
                ultraviolet::Vec2::new(0.0, -2.0),
            ),
            (
                ultraviolet::Vec2::new(0.0, -1.0),
                ultraviolet::Vec2::new(0.0, -2.0),
                ultraviolet::Vec2::new(-1.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(0.0, -2.0),
                ultraviolet::Vec2::new(-1.0, -1.0),
                ultraviolet::Vec2::new(-2.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(-1.0, -1.0),
                ultraviolet::Vec2::new(-2.0, -1.0),
                ultraviolet::Vec2::new(-2.0, -2.0),
            ),
            (
                ultraviolet::Vec2::new(-2.0, -1.0),
                ultraviolet::Vec2::new(-2.0, -2.0),
                ultraviolet::Vec2::new(-3.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(-2.0, -2.0),
                ultraviolet::Vec2::new(-3.0, -1.0),
                ultraviolet::Vec2::new(-3.0, 0.0),
            ),
            (
                ultraviolet::Vec2::new(-3.0, -1.0),
                ultraviolet::Vec2::new(-3.0, 0.0),
                ultraviolet::Vec2::new(-4.0, 0.0),
            ),
            (
                ultraviolet::Vec2::new(-3.0, 0.0),
                ultraviolet::Vec2::new(-4.0, 0.0),
                ultraviolet::Vec2::new(-3.0, 1.0),
            ),
            (
                ultraviolet::Vec2::new(-4.0, 0.0),
                ultraviolet::Vec2::new(-3.0, 1.0),
                ultraviolet::Vec2::new(-2.0, 1.0),
            ),
            (
                ultraviolet::Vec2::new(-3.0, 1.0),
                ultraviolet::Vec2::new(-2.0, 1.0),
                ultraviolet::Vec2::new(-2.0, 2.0),
            ),
            (
                ultraviolet::Vec2::new(-2.0, 1.0),
                ultraviolet::Vec2::new(-2.0, 2.0),
                ultraviolet::Vec2::new(-1.0, 1.0),
            ),
            (
                ultraviolet::Vec2::new(-2.0, 2.0),
                ultraviolet::Vec2::new(-1.0, 1.0),
                ultraviolet::Vec2::new(-1.0, 0.0),
            ),
            (
                ultraviolet::Vec2::new(-1.0, 1.0),
                ultraviolet::Vec2::new(-1.0, 0.0),
                ultraviolet::Vec2::new(0.0, 0.0),
            ),
            (
                ultraviolet::Vec2::new(-1.0, 0.0),
                ultraviolet::Vec2::new(0.0, 0.0),
                ultraviolet::Vec2::new(1.0, 0.0),
            ),
        ],
        DemoType::Castro => vec![
            (
                ultraviolet::Vec2::new(0.0, 0.0),
                ultraviolet::Vec2::new(1.0, 0.0),
                ultraviolet::Vec2::new(1.0, 1.0),
            ),
            (
                ultraviolet::Vec2::new(1.0, 0.0),
                ultraviolet::Vec2::new(1.0, 1.0),
                ultraviolet::Vec2::new(2.0, 1.0),
            ),
            (
                ultraviolet::Vec2::new(1.0, 1.0),
                ultraviolet::Vec2::new(2.0, 1.0),
                ultraviolet::Vec2::new(2.0, 2.0),
            ),
            (
                ultraviolet::Vec2::new(2.0, 1.0),
                ultraviolet::Vec2::new(2.0, 2.0),
                ultraviolet::Vec2::new(3.0, 1.0),
            ),
            (
                ultraviolet::Vec2::new(2.0, 2.0),
                ultraviolet::Vec2::new(3.0, 1.0),
                ultraviolet::Vec2::new(3.0, 0.0),
            ),
            (
                ultraviolet::Vec2::new(3.0, 1.0),
                ultraviolet::Vec2::new(3.0, 0.0),
                ultraviolet::Vec2::new(4.0, 0.0),
            ),
            (
                ultraviolet::Vec2::new(3.0, 0.0),
                ultraviolet::Vec2::new(4.0, 0.0),
                ultraviolet::Vec2::new(3.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(4.0, 0.0),
                ultraviolet::Vec2::new(3.0, -1.0),
                ultraviolet::Vec2::new(2.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(3.0, -1.0),
                ultraviolet::Vec2::new(2.0, -1.0),
                ultraviolet::Vec2::new(2.0, -2.0),
            ),
            (
                ultraviolet::Vec2::new(2.0, -1.0),
                ultraviolet::Vec2::new(2.0, -2.0),
                ultraviolet::Vec2::new(1.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(2.0, -2.0),
                ultraviolet::Vec2::new(1.0, -1.0),
                ultraviolet::Vec2::new(0.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(1.0, -1.0),
                ultraviolet::Vec2::new(0.0, -1.0),
                ultraviolet::Vec2::new(0.0, -2.0),
            ),
            (
                ultraviolet::Vec2::new(0.0, -1.0),
                ultraviolet::Vec2::new(0.0, -2.0),
                ultraviolet::Vec2::new(-1.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(0.0, -2.0),
                ultraviolet::Vec2::new(-1.0, -1.0),
                ultraviolet::Vec2::new(-2.0, -2.0),
            ),
            (
                ultraviolet::Vec2::new(-1.0, -1.0),
                ultraviolet::Vec2::new(-2.0, -2.0),
                ultraviolet::Vec2::new(-2.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(-2.0, -2.0),
                ultraviolet::Vec2::new(-2.0, -1.0),
                ultraviolet::Vec2::new(-3.0, -1.0),
            ),
            (
                ultraviolet::Vec2::new(-2.0, -1.0),
                ultraviolet::Vec2::new(-3.0, -1.0),
                ultraviolet::Vec2::new(-4.0, 0.0),
            ),
            (
                ultraviolet::Vec2::new(-3.0, -1.0),
                ultraviolet::Vec2::new(-4.0, 0.0),
                ultraviolet::Vec2::new(-3.0, 0.0),
            ),
            (
                ultraviolet::Vec2::new(-4.0, 0.0),
                ultraviolet::Vec2::new(-3.0, 0.0),
                ultraviolet::Vec2::new(-3.0, 1.0),
            ),
            (
                ultraviolet::Vec2::new(-3.0, 0.0),
                ultraviolet::Vec2::new(-3.0, 1.0),
                ultraviolet::Vec2::new(-2.0, 2.0),
            ),
            (
                ultraviolet::Vec2::new(-3.0, 1.0),
                ultraviolet::Vec2::new(-2.0, 2.0),
                ultraviolet::Vec2::new(-2.0, 1.0),
            ),
            (
                ultraviolet::Vec2::new(-2.0, 2.0),
                ultraviolet::Vec2::new(-2.0, 1.0),
                ultraviolet::Vec2::new(-1.0, 1.0),
            ),
            (
                ultraviolet::Vec2::new(-2.0, 1.0),
                ultraviolet::Vec2::new(-1.0, 1.0),
                ultraviolet::Vec2::new(-1.0, 0.0),
            ),
            (
                ultraviolet::Vec2::new(-1.0, 1.0),
                ultraviolet::Vec2::new(-1.0, 0.0),
                ultraviolet::Vec2::new(0.0, 0.0),
            ),
            (
                ultraviolet::Vec2::new(-1.0, 0.0),
                ultraviolet::Vec2::new(0.0, 0.0),
                ultraviolet::Vec2::new(1.0, 0.0),
            )
        ]
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 9)]
async fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();

    let config = quarkstrom::Config {
        window_mode: quarkstrom::WindowMode::Windowed(1280, 720),
    };

    let tps_cap: Option<u32> = None;

    let desired_frame_time =
        tps_cap.map(|tps| std::time::Duration::from_secs_f64(1.0 / tps as f64));

    let mut threader = Threader::new(8, 8192).await;

    // let mut simulation = Simulation::new(65536, None);

    threader.init(&rt);

    // tokio::spawn(async move {
    //     loop {
    //         let frame_timer = std::time::Instant::now();
    //
    //         threader.update();
    //         // simulation.update_simulation(0);
    //
    //         // Cap tps
    //         if let Some(desired_frame_time) = desired_frame_time {
    //             while frame_timer.elapsed() < desired_frame_time {}
    //         }
    //
    //         if *STOPPED.lock() {
    //             break;
    //         }
    //     }
    // });

    quarkstrom::run::<Renderer>(config);
    *STOPPED.lock() = true;
}

#[derive(Clone)]
enum DemoType {
    Tarkosky,
    Castro
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
            scale: 100.0,
            demo_type: DEMO_TYPE.lock().to_owned(),
        }
    }

    fn input(&mut self, input: &WinitInputHelper, _width: u16, _height: u16) {
        if input.held_shift() {
            self.pos = Vec2::new(80.0, 0.0);
            self.scale = 10.0;
        } else {
            self.pos = Vec2::new(0.0, 0.0);
            self.scale = 100.0;
        }
    }

    fn render(&mut self, ctx: &mut quarkstrom::RenderContext) {
        ctx.set_view_pos(ultraviolet::Vec2::new(self.pos.x as f32, self.pos.y as f32));
        ctx.set_view_scale(self.scale);

        if let Some(particles) = PARTICLES.lock().clone() {
            ctx.clear_circles();
            ctx.clear_lines();

            for particle_vec in particles {
                for particle in particle_vec {
                    // https://www.rapidtables.com/convert/color/hsl-to-rgb.html
                    let h = particle.angle / std::f64::consts::PI * 180.0;
                    let s = 1.0f64;
                    let l = 0.5f64;

                    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
                    let x = c * (1.0 - (h / 60.0 % 2.0 - 1.0).abs());
                    let m = l - c / 2.0;

                    let (rp, gp, bp): (f64, f64, f64) = {
                        let h = particle.angle / std::f64::consts::PI * 180.0;
                        if h < 60.0 {
                            (c, x, 0.0)
                        } else if h < 120.0 {
                            (x, c, 0.0)
                        } else if h < 180.0 {
                            (0.0, c, x)
                        } else if h < 240.0 {
                            (0.0, x, c)
                        } else if h < 300.0 {
                            (x, 0.0, c)
                        } else {
                            (c, 0.0, x)
                        }
                    };

                    let (r, g, b) = ((rp + m) * 255.0, (gp + m) * 255.0, (bp + m) * 255.0);

                    ctx.draw_circle(
                        ultraviolet::Vec2::new(particle.pos.x as f32, particle.pos.y as f32),
                        0.1,
                        [r as u8, g as u8, b as u8, 255],
                    );
                }
            }
        }

        let points = get_lines(&self.demo_type);

        for i in 0..points.len() {
            ctx.draw_line(points[i].0 * 40.0, points[i].1 * 40.0, [255, 255, 255, 255]);
        }

        match self.demo_type {
            DemoType::Tarkosky => {
                ctx.draw_circle(ultraviolet::Vec2::new(80.0, 0.0), 0.2, [255, 0, 0, 128]);
            },
            DemoType::Castro => {
                ctx.draw_circle(ultraviolet::Vec2::new(80.0, 0.0), 0.2, [255, 0, 0, 128]);
            }
        }
    }

    fn gui(&mut self, _ctx: &egui::Context) {
        return;
    }
}

fn dst(p1: Vec2, p2: Vec2) -> f64 {
    ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt()
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

// https://www.jeffreythompson.org/collision-detection/line-point.php
fn line_point(p1: Vec2, p2: Vec2, p3: Vec2) -> bool {
    let d1 = ((p3.x - p1.x).powi(2) + (p3.y - p1.y).powi(2)).sqrt();
    let d2 = ((p3.x - p2.x).powi(2) + (p3.y - p2.y).powi(2)).sqrt();

    let len = ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt();

    if d1 + d2 >= len - 0.1 && d1 + d2 <= len + 0.1 {
        return true;
    }
    false
}

#[derive(Clone)]
struct Particle {
    pos: Vec2,
    angle: f64,
}

#[derive(Clone)]
struct Simulation {
    particles: Vec<Particle>
}

impl Simulation {
    fn new(num_particles: usize, init_particles: Option<Vec<Particle>>) -> Self {
        let mut particles = Vec::new();
        if let Some(ptcls) = init_particles {
            particles = ptcls
        } else {
            for i in 0..num_particles {
                particles.push(Particle {
                    pos: Vec2::new(-80.0, 0.0),
                    angle: std::f64::consts::TAU * (i as f64 / num_particles as f64),
                });
            }
        }

        Self {
            particles
        }
    }

    fn update_simulation(&mut self, id: usize) {
        for particle in &mut self.particles {
            let mut goal_pos =
                particle.pos + Vec2::new(particle.angle.cos(), particle.angle.sin()) * 0.1;

            let points = get_lines(&DEMO_TYPE.lock());

            for i in 0..points.len() {
                if let Some(intersection_point) = line_line(
                    particle.pos,
                    goal_pos,
                    Vec2::new(points[i].0.x as f64, points[i].0.y as f64) * 40.0,
                    Vec2::new(points[i].1.x as f64, points[i].1.y as f64) * 40.0,
                ) {
                    let remaining_dist = 0.1 - dst(particle.pos, intersection_point);

                    particle.pos = intersection_point;

                    // lmaoooooo i used desmos to find this xd i don't know how it works
                    // https://www.desmos.com/calculator/popf3ryijo

                    // i couldn't find out how to get the normal without just arctan
                    let normal = ((points[i].0.x as f64 - points[i].1.x as f64)
                        / (points[i].0.y as f64 - points[i].1.y as f64))
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
                        + Vec2::new(particle.angle.cos(), particle.angle.sin()) * remaining_dist;
                } else if line_point(
                    particle.pos,
                    goal_pos,
                    Vec2::new(points[i].1.x as f64, points[i].1.y as f64) * 40.0,
                ) {
                    let intersection_point = Vec2::new(points[i].1.x as f64, points[i].1.y as f64) * 40.0;

                    let remaining_dist = 0.1 - dst(particle.pos, intersection_point);

                    particle.pos = intersection_point;

                    // lmaoooooo i used desmos to find this xd i don't know how it works
                    // https://www.desmos.com/calculator/popf3ryijo

                    // i couldn't find out how to get the normal without just arctan
                    let normal = ((points[i].0.x as f64 - points[i].2.x as f64)
                        / (points[i].0.y as f64 - points[i].2.y as f64))
                        .atan() + std::f64::consts::FRAC_PI_2;
                    let mut normal_vec = Vec2::new(normal.sin(), normal.cos());

                    let dir_vec = (goal_pos - particle.pos).normalized();

                    if dir_vec.dot(normal_vec) < 0.0 {
                        normal_vec = -normal_vec;
                    }

                    let reflected = dir_vec - 2.0 * dir_vec.dot(normal_vec) * normal_vec;

                    particle.angle = f64::atan2(reflected.y, reflected.x);

                    goal_pos = intersection_point
                        + Vec2::new(particle.angle.cos(), particle.angle.sin()) * remaining_dist;
                }
            }

            particle.pos = goal_pos;
        }

        *PARTICLES.lock().as_mut().unwrap().get_mut(id).unwrap() = self.particles.clone();
    }

    fn register_manager(mut self, id: usize, rt: &tokio::runtime::Runtime) {
        rt.spawn(async move {
            loop {
                self.update_simulation(id);

                if *STOPPED.lock() {
                    break;
                }
            }
        });
    }
}

struct Threader {
    simulations: Vec<Simulation>
}

impl Threader {
    async fn new(sims: usize, particles_per_sim: usize) -> Self {
        let mut simulations: Vec<Simulation> = Vec::new();

        let mut global_particles: Vec<Vec<Particle>> = Vec::new();

        for sim in 0..sims {
            global_particles.push(Vec::new());

            *PARTICLES.lock() = Some(global_particles.clone());

            let mut particles = Vec::new();
            for i in 0..particles_per_sim {
                particles.push(Particle {
                    pos: Vec2::new(-80.0, 0.0),
                    angle: std::f64::consts::TAU * ((i + sim * particles_per_sim) as f64 / (particles_per_sim * sims) as f64),
                });
            }

            simulations.push(Simulation::new(particles_per_sim, Some(particles)));
        }

        Self {
            simulations
        }
    }

    fn init(&mut self, rt: &tokio::runtime::Runtime) {
        for i in 0..self.simulations.len() {
            let sim = self.simulations[i].clone();
            sim.register_manager(i, &rt);
        }
    }
}