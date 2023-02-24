{
    const canvas = document.querySelector("#dist-canvas");
    const offscreen = canvas.transferControlToOffscreen();

    const sliderIds = ['xoff', 'yoff', 'sigma', 'excentricity', 'angle'];
    const sliders = Object.fromEntries(
        sliderIds.map(id => [id, document.getElementById(id)])
    );

    const worker = new Worker("darts_worker.js");
    worker.postMessage({
        canvas: offscreen,
        sliderValues: Object.fromEntries(
            sliderIds.map(id => [id, sliders[id].value])
        )
    }, [offscreen]);

    sliderIds.map(id => sliders[id].addEventListener("input", () => {
        worker.postMessage({
            sliderValues: {[id]: parseFloat(sliders[id].value)}
        });
    }));
}
