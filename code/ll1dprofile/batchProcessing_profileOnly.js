const fs = require("fs");
const { JSDOM } = require("jsdom");

async function loadD3() {
  const d3 = await import("d3");
    const directoryPath = "../basin_points/";
    fs.readdir(directoryPath, function (err, files) {
    if (err) {
      return console.log("Error reading directory: " + err);
    }
    // Process each CSV file
    files.forEach(function (file) {
      if (file.endsWith(".json")) {
        // Read CSV data
        const mergeTreeData = fs.readFileSync(directoryPath + file, "utf-8");
        console.log(`Processing ${file}`);

        const data = JSON.parse(mergeTreeData);
        const points = data.points;
        console.log(points);

        // Create a virtual DOM for D3.js
        const dom = new JSDOM(
          `<!DOCTYPE html><body><div id="visualization"></div></body>`,
          {
            pretendToBeVisual: true,
          },
        );
        global.document = dom.window.document;

        let body = d3.select(dom.window.document.querySelector("body"));

        // drawing code
        // Declare the chart dimensions and margins.
        const width = 940;
        const height = 600;
        const marginTop = 20;
        const marginRight = 20;
        const marginBottom = 30;
        const marginLeft = 40;
        const backgroundColor = "#efefef";

        const svg = body
          .select("#visualization")
          .append("svg")
          .attr("width", width)
          .attr("height", height);

        /**
         * DRAWING the plots using D3
         */

        const loss_max = d3.max(points, (p) => d3.max(p.area, (d) => d[1]));
        const loss_min = d3.min(points, (p) => d3.min(p.area, (d) => d[1]));
        const x_max = d3.max(points, (p) => d3.max(p.area, (d) => d[0]));
        const x_min = d3.min(points, (p) => d3.min(p.area, (d) => d[0]));

        const xScale = d3
          .scaleLinear()
          .domain([x_min, x_max])
          .range([marginLeft, width - marginRight]);

        const yScale = d3
          .scaleLinear()
          .domain([loss_min + 0.00000001, loss_max])
          .range([height - marginBottom, marginTop]);

        const basinColors = d3
          .scaleSequential()
          .domain([loss_max, loss_min])
          .interpolator(d3.interpolateBlues);

        const line = d3
          .line()
          .x((d) => xScale(d[0]))
          .y((d) => yScale(d[1]));

        // draw rectangle background
        svg
          .append("rect")
          .attr("x", marginLeft)
          .attr("y", marginTop)
          .attr("width", width - marginRight - marginLeft)
          .attr("height", height - marginBottom - marginTop)
          .attr("stroke", backgroundColor)
          .attr("fill", backgroundColor)
          .attr("opacity", 1.0);

        svg
          .selectAll(".line")
          .data(points)
          .join("path")
          .attr("class", "line")
          .attr("d", (d) => line(d.area))
          .attr("stroke", (d) => basinColors(d3.mean(d.area, (dd) => dd[1])))
          .attr("stroke-width", 1.5)
          .attr("fill", (d) => basinColors(d3.mean(d.area, (dd) => dd[1])))
          .attr("opacity", 1);

        // svg
        //   .append("g")
        //   .attr("transform", `translate(0,${height - marginBottom})`)
        //   .call(d3.axisBottom(xScale));

        // Add the y-axis.
        // svg
        //   .append("g")
        //   .attr("transform", `translate(${marginLeft},0)`)
        //   .call(d3.axisLeft(yScale));

        // svg
        //   .append("text")
        //   .attr("x", 15)
        //   .attr("y", 15)
        //   .attr("text-anchor", "start")
        //   .text("Loss");
        fs.writeFileSync(
          `./profileOnly_output/${file.replace(".json", ".html")}`,
          body.html(),
        );
        console.log(`Visualization saved for ${file}`);
      }
    });
  });
}

// Call the async function to load D3.js
loadD3().catch((error) => {
  console.error("Error loading D3.js:", error);
});
