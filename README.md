# Billiards Sim
###### I kind of departed from the fall theme, but uh, the particles are falling in the direction that they have velocity in? 😖

Made for Hack Club Siege Week 4 with the theme "fall"!

So, my initial idea was to make one of those falling particle simulation things after watching
[this](https://www.youtube.com/watch?v=vG0ina57osc) video, and so took the code from another particle simulation thing
I've seen, [Quarkstrom](https://github.com/DeadlockCode/quarkstrom), and [forked](https://github.com/Omay238/quarkstrom)
it to update the libraries, and add some features. That took around 11.5 hours, while the actual simulation (this
repository) took around 8 hours as of writing this readme.

This was pretty difficult to make, as there were a lot of things that I had never done with Rust, and I'm still quite
new. It would've been quite reasonable to do this in like half the time, if I were to know what I was doing, but
unfortunately, I do not. There's also some bugs that I have not been able to squash. (Who knows if this'll change by the
time I submit the project)

## Running

The easiest way to run is to download a release from GitHub (only time will tell if I get that working in time lol), but
if that isn't available, you can of course run it from source.

First, install [rustup](https://rustup.rs/)

Then, either clone with `git clone --recursive https://github.com/Omay238/particlesim.git` or download a ZIP from 
`Code > Download ZIP` on the top of the page. If you download a zip, then you also need to download the Quarkstrom
source from [here](https://github.com/Omay238/quarkstrom/tree/master), and extract it to where the Quarkstrom directory
in this project's code is located.

Finally, navigate into the directory in your terminal of choice and run `cargo run --release` to run!

Press shift to zoom into a point of interest (if applicable), and escape to close the program.

## Demo

&lt;put this in, future me!&gt;