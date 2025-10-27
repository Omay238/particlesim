use quarkstrom;

use egui;
use tokio;
use ultraviolet::DVec2 as Vec2;
use winit_input_helper::WinitInputHelper;

use once_cell::sync::Lazy;

// Used to communicate between the simulation and renderer threads
// all the particles from the simulations so the renderer can see it
static PARTICLES: Lazy<egui::mutex::Mutex<Option<Vec<Vec<Particle>>>>> =
    Lazy::new(|| egui::mutex::Mutex::new(vec![vec![]].into()));
// self-explanatory. is it stopped?
static STOPPED: Lazy<egui::mutex::Mutex<bool>> = Lazy::new(|| egui::mutex::Mutex::new(false));
// which of the whopping 2 maps is it using
static DEMO_TYPE: Lazy<egui::mutex::Mutex<DemoType>> = Lazy::new(|| egui::mutex::Mutex::new(DemoType::Castro));
// amount of steps into the simulation
static STEPS: Lazy<egui::mutex::Mutex<usize>> = Lazy::new(|| egui::mutex::Mutex::new(0));
// this is the number of threads
static THREADS: Lazy<egui::mutex::Mutex<usize>> = Lazy::new(|| egui::mutex::Mutex::new(8));

// this function lowkey sucks really hard lmao
// i'd refactor it but i'm tired
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

// wow it's two asyncs
#[tokio::main]
async fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(*THREADS.lock() + 1)
        .enable_all()
        .build()
        .unwrap();

    let config = quarkstrom::Config {
        window_mode: quarkstrom::WindowMode::Windowed(1280, 720),
    };

    let tps_cap: Option<u32> = Some(60);

    let desired_frame_time =
        tps_cap.map(|tps| std::time::Duration::from_secs_f64(1.0 / tps as f64));

    let mut threader = Threader::new(*THREADS.lock(), 8192).await;
    threader.init(desired_frame_time, &rt);

    // let mut simulation = Simulation::new(65536, None);
    // tokio::spawn(async move {
    //     loop {
    //         let frame_timer = std::time::Instant::now();
    //
    //         simulation.update_simulation(0);
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

#[derive(Clone, Debug)]
#[derive(PartialEq)]
enum DemoType {
    Tarkosky,
    Castro
}

#[derive(Clone)]
struct Renderer {
    pos: Vec2,
    scale: f32
}

impl quarkstrom::Renderer for Renderer {
    fn new() -> Self {
        Self {
            pos: Vec2::new(0.0, 0.0),
            scale: 100.0
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
                    let h = (particle.heading.y.atan2(particle.heading.x) + std::f64::consts::PI) * 180.0 / std::f64::consts::PI;
                    let s = 1.0f64;
                    let l = 0.5f64;

                    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
                    let x = c * (1.0 - (h / 60.0 % 2.0 - 1.0).abs());
                    let m = l - c / 2.0;

                    let (rp, gp, bp): (f64, f64, f64) = {
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

        // symbols xd
        let points = get_lines(&*DEMO_TYPE.lock());

        for i in 0..points.len() {
            ctx.draw_line(points[i].0 * 40.0, points[i].1 * 40.0, [255, 255, 255, 255]);
        }

        match *DEMO_TYPE.lock() {
            DemoType::Tarkosky => {
                ctx.draw_circle(ultraviolet::Vec2::new(80.0, 0.0), 0.2, [255, 0, 0, 128]);
            },
            DemoType::Castro => {
                ctx.draw_circle(ultraviolet::Vec2::new(80.0, 0.0), 0.2, [255, 0, 0, 128]);
            }
        }
    }

    fn gui(&mut self, ctx: &egui::Context) {
        egui::Window::new("Billiards").show(&ctx, |ui| {
            let before = DEMO_TYPE.lock().clone();
            egui::ComboBox::from_label("Boundaries")
                .selected_text(format!("{:?}", *DEMO_TYPE.lock()))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut *DEMO_TYPE.lock(), DemoType::Tarkosky, "Tarkosky");
                    ui.selectable_value(&mut *DEMO_TYPE.lock(), DemoType::Castro, "Castro");
                }).response;

            if before != *DEMO_TYPE.lock() {
                *PARTICLES.lock() = vec![vec![]; *THREADS.lock()].into();
            }
        });
    }
}

fn dst(p1: Vec2, p2: Vec2) -> f64 {
    ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt()
}

// https://www.jeffreythompson.org/collision-detection/line-line.php
fn line_line(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> Option<Vec2> {
    // how does this work? i haven't a clue
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
#[derive(PartialEq)]
struct Particle {
    pos: Vec2,
    heading: Vec2,
}

#[derive(Clone)]
struct Simulation {
    particles: Vec<Particle>,
    initial_particles: Vec<Particle>,
    steps: usize
}

impl Simulation {
    fn new(num_particles: usize, init_particles: Option<Vec<Particle>>) -> Self {
        let mut particles = Vec::new();
        if let Some(ptcls) = init_particles {
            particles = ptcls
        } else {
            for i in 0..num_particles {
                let angle = std::f64::consts::TAU * (i as f64 / num_particles as f64);
                particles.push(Particle {
                    pos: Vec2::new(-80.0, 0.0),
                    heading: Vec2::new(angle.cos(), angle.sin()),
                });
            }
        }

        Self {
            particles: particles.clone(),
            initial_particles: particles,
            steps: 0
        }
    }

    fn update_simulation(&mut self, id: usize) {
        // uhh, reset particles when particles are reset
        if *PARTICLES.lock().as_mut().unwrap().get_mut(id).unwrap() != self.particles {
            self.particles = self.initial_particles.clone();
        }

        for particle in &mut self.particles {
            let mut goal_pos =
                particle.pos + particle.heading * 0.1;

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
                    // make the normal into a vector
                    let mut normal_vec = Vec2::new(normal.sin(), normal.cos());

                    // where is the particle going?
                    let dir_vec = (goal_pos - particle.pos).normalized();

                    // make sure the normal is in the right direction
                    if dir_vec.dot(normal_vec) > 0.0 {
                        normal_vec = -normal_vec;
                    }

                    // reflect the particle across the normal
                    let reflected = dir_vec - 2.0 * dir_vec.dot(normal_vec) * normal_vec;

                    particle.heading = reflected;

                    goal_pos = intersection_point
                        + particle.heading * remaining_dist;
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
                        .atan()
                        + std::f64::consts::FRAC_PI_2;
                    // make the normal into a vector
                    let mut normal_vec = Vec2::new(normal.sin(), normal.cos());

                    // where is the particle going?
                    let dir_vec = (goal_pos - particle.pos).normalized();

                    // make sure the normal is in the right direction
                    if dir_vec.dot(normal_vec) > 0.0 {
                        normal_vec = -normal_vec;
                    }

                    // reflect the particle across the normal
                    let reflected = dir_vec - 2.0 * dir_vec.dot(normal_vec) * normal_vec;

                    particle.heading = reflected;

                    goal_pos = intersection_point
                        + particle.heading * remaining_dist;
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

                self.steps += 1;

                // keep locked to global thread
                while *STEPS.lock() < self.steps {}

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
                let angle = std::f64::consts::TAU * ((i + sim * particles_per_sim) as f64 / (particles_per_sim * sims) as f64);
                particles.push(Particle {
                    pos: Vec2::new(-80.0, 0.0),
                    heading: Vec2::new(angle.cos(), angle.sin()),
                });
            }

            simulations.push(Simulation::new(particles_per_sim, Some(particles)));
        }

        Self {
            simulations
        }
    }

    fn init(&mut self, desired_frame_time: Option<std::time::Duration>, rt: &tokio::runtime::Runtime) {
        for i in 0..self.simulations.len() {
            let sim = self.simulations[i].clone();
            // create all of the subprocesses
            sim.register_manager(i, &rt);
        }
        rt.spawn(async move {
            loop {
                let frame_timer = std::time::Instant::now();

                // make all the threads locked to the speed of a central thread
                *STEPS.lock() += 1;

                // Cap tps
                if let Some(desired_frame_time) = desired_frame_time {
                    while frame_timer.elapsed() < desired_frame_time {}
                }
            }
        });
    }
}