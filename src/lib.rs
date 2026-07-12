use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray4};
use crossbeam::channel::{bounded, Receiver, Sender};
use rayon::ThreadPoolBuilder;
use ndarray::Array4;
use ffmpeg_next as ffmpeg;
use ffmpeg::format::{input, Pixel};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context as ScalerContext, flag::Flags};

pub struct FrameBatch {
    pub data: Vec<u8>,
    pub batch_size: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

#[pyclass]
pub struct VideoLoader {
    receiver: Receiver<FrameBatch>,
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
}

#[pymethods]
impl VideoLoader {
    #[new]
    pub fn new(video_path: String, batch_size: usize, height: usize, width: usize) -> Self {
        // Initialize FFmpeg globally
        ffmpeg::init().expect("Failed to initialize ffmpeg");
        
        // Lock-free queue
        let (sender, receiver) = bounded::<FrameBatch>(4);
        let channels = 3;

        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        
        pool.spawn(move || {
            decode_loop(video_path, sender, batch_size, height, width, channels);
        });

        VideoLoader {
            receiver,
            batch_size,
            channels,
            height,
            width,
        }
    }

    pub fn next_batch<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py PyArray4<u8>>> {
        py.allow_threads(|| {
            self.receiver.recv().ok()
        }).map(|batch| {
            // For packed RGB24, the memory layout is (batch_size, height, width, channels)
            let dims = (batch.batch_size, batch.height, batch.width, batch.channels);
            let array = Array4::from_shape_vec(dims, batch.data).unwrap();
            array.into_pyarray(py)
        })
        .map_or(Ok(None), |arr| Ok(Some(arr)))
    }
}

fn decode_loop(
    video_path: String, 
    sender: Sender<FrameBatch>, 
    batch_size: usize, 
    out_height: usize, 
    out_width: usize, 
    channels: usize
) {
    let mut ictx = input(&video_path).expect("Could not open video file");
    let input_stream = ictx.streams().best(Type::Video).expect("No video stream found");
    let video_stream_index = input_stream.index();
    
    let context_decoder = ffmpeg::codec::context::Context::from_parameters(input_stream.parameters()).unwrap();
    let mut decoder = context_decoder.decoder().video().unwrap();
    
    let mut scaler = ScalerContext::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        Pixel::RGB24, // Hardware accelerated colorspace conversion
        out_width as u32,
        out_height as u32,
        Flags::FAST_BILINEAR,
    ).unwrap();

    let frame_size = channels * out_height * out_width;
    let batch_bytes = batch_size * frame_size;
    
    let mut buffer: Vec<u8> = Vec::with_capacity(batch_bytes);
    let mut frames_collected = 0;

    let mut decoded_frame = ffmpeg::frame::Video::empty();
    let mut rgb_frame = ffmpeg::frame::Video::new(Pixel::RGB24, out_width as u32, out_height as u32);
    
    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            if decoder.send_packet(&packet).is_err() { continue; }
            
            while decoder.receive_frame(&mut decoded_frame).is_ok() {
                scaler.run(&decoded_frame, &mut rgb_frame).unwrap();
                
                let data = rgb_frame.data(0);
                let linesize = rgb_frame.stride(0) as usize;
                let width_bytes = out_width * channels;
                
                // Copy frame scanlines into contiguous buffer
                for y in 0..out_height {
                    let start = y * linesize;
                    let end = start + width_bytes;
                    buffer.extend_from_slice(&data[start..end]);
                }
                
                frames_collected += 1;
                
                if frames_collected == batch_size {
                    let batch = FrameBatch {
                        data: buffer,
                        batch_size,
                        channels,
                        height: out_height,
                        width: out_width,
                    };
                    
                    if sender.send(batch).is_err() {
                        return; // Python receiver dropped, gracefully terminate thread
                    }
                    
                    buffer = Vec::with_capacity(batch_bytes);
                    frames_collected = 0;
                }
            }
        }
    }
}

#[pymodule]
fn tensorvid(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VideoLoader>()?;
    Ok(())
}
