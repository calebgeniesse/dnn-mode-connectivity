const fs = require("fs");
const { JSDOM } = require("jsdom");

async function loadD3() {
  const d3 = await import("d3");
  const directoryPath = "../branch_decomposition/branch_decomposition_results/";
  fs.readdir(directoryPath, function (err, files) {
    if (err) {
      return console.log("Error reading directory: " + err);
    }
    // Process each CSV file
    files.forEach(function (file) {
      if (file.endsWith("branch_decomposition.csv")) {
        // Read CSV data
        const mergeTreeData = fs.readFileSync(directoryPath + file, "utf-8");
        console.log(`Processing ${file}`);

        console.log("edge_data");
        // console.log(mergeTreeData);
        const edge_data = mergeTreeData.split("\r\n");
        let edgesList = [];
        edge_data.forEach((el) => {
          const edge = el.split(",");
          if (edge.length > 1) {
            edgesList.push(edge.map(Number));
          }
        });
        // console.log(edgesList);

        const nodeDictFile = file.replace("branch_decomposition", "node_dict");
        const nodeDictData = fs.readFileSync(
          directoryPath + nodeDictFile,
          "utf-8",
        );
        console.log("node_dict_data");
        // console.log(nodeDictData);
        let nodesLoss = [];
        const node_dict_data = nodeDictData.split("\r\n");
        node_dict_data.forEach((el) => {
          const node = el.split(",");
          if (node.length > 1) {
            nodesLoss.push(node.map(Number));
          }
        });

        console.log(nodesLoss);

        // console.log("node_dict_data");
        // console.log(nodeDictData);

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
        const width = 1640;
        const height = 700;
        const marginTop = 30;
        const marginRight = 30;
        const marginBottom = 30;
        const marginLeft = 50;

        // test edge data
        // edgesList = [
        //   [0, 3, 4, 5],
        //   [1, 3],
        //   [2, 4],
        // ];
        //
        // nodesLoss = [
        //   [0, 0],
        //   [1, 1],
        //   [2, 43],
        //   [3, 46],
        //   [4, 48],
        //   [5, 100],
        // ];

        const edges = [];
        const nodes = {};
        edgesList.forEach((el, branchID) => {
          for (let i = 0; i < el.length - 1; i++) {
            edges.push([el[i], el[i + 1], branchID]);
            nodes[el[i]] = {
              branchID: branchID,
              loss: nodesLoss[el[i]][1],
            };
            if (branchID === 0) {
              nodes[el[i + 1]] = {
                branchID: branchID,
                loss: nodesLoss[el[i + 1]][1],
              };
            }
          }
        });

        /**
         * DRAWING the plots using D3
         */
        const xScale = d3
          .scalePoint()
          .domain(edgesList.map((d) => d[0]).sort((a, b) => a - b))
          .range([marginLeft + 30, width - marginRight]);

        const yScale = d3
          .scaleLinear()
          .domain(d3.extent(nodesLoss, (d) => d[1]))
          .range([height - marginBottom - 30, marginTop]);

        const color = d3
          .scaleOrdinal(d3.schemeTableau10)
          .domain(xScale.domain());
        // Draw
        const svg = body
          .select("#visualization")
          .append("svg")
          .attr("width", width)
          .attr("height", height);

        svg
          .selectAll(".line")
          .data(edges)
          .join("line")
          .attr("class", "line")
          .attr("x1", (d) => xScale(nodes[d[0]].branchID))
          .attr("x2", (d) => xScale(nodes[d[1]].branchID))
          .attr("y1", (d) => yScale(nodes[d[0]].loss))
          .attr("y2", (d) => yScale(nodes[d[1]].loss))
          .attr("stroke", (d) => color(d[2]))
          .attr("stroke-width", "3");

        svg
          .selectAll("circle")
          .data(nodesLoss)
          .join("circle")
          .attr("fill", (d) => {
            if (d.type === "main") {
              return "#fff";
            } else {
              return "#fff";
            }
          })
          .attr("stroke", "#666")
          .attr("stroke-width", 1.5)
          .attr("cx", (d) => xScale(nodes[d[0]].branchID))
          .attr("cy", (d) => yScale(nodes[d[0]].loss))
          .attr("r", 4);

        // svg
        //   .selectAll("text")
        //   .data(nodesLoss)
        //   .join("text")
        //   .attr("fill", (d) => {
        //     if (d.type === "main") {
        //       return "black";
        //     } else {
        //       return "black";
        //     }
        //   })
        //   .attr("x", (d) => xScale(nodes[d[0]].branchID) + 15)
        //   .attr("y", (d) => yScale(nodes[d[0]].loss) + 5)
        //   .attr("font-size", 18)
        //   .text((d) => d[0]);

        // Add the x-axis.
        svg
          .append("g")
          .attr("transform", `translate(0,${height - marginBottom})`)
          .call(d3.axisBottom(xScale))
          .selectAll("text")
          .attr("font-size", 14);

        // Add the y-axis.
        svg
          .append("g")
          .attr("transform", `translate(${marginLeft},0)`)
          .call(d3.axisLeft(yScale))
          .selectAll("text")
          .attr("font-size", 14);

        svg
          .append("text")
          .attr("x", 15)
          .attr("y", 15)
          .attr("text-anchor", "start")
          .text("Loss");

        svg
          .append("text")
          .attr("x", width - 15)
          .attr("y", height - 35)
          .attr("text-anchor", "end ")
          .text("Branch ID");

        fs.writeFileSync(
          `./output/${file.replace(".csv", ".html")}`,
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
