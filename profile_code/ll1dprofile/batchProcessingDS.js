const fs = require("fs");
const { JSDOM } = require("jsdom");

async function loadD3() {
  const d3 = await import("d3");
  const directoryPath = "../paraview_files/";
  fs.readdir(directoryPath, function (err, files) {
    if (err) {
      return console.log("Error reading directory: " + err);
    }
    // Process each CSV file
    files.forEach(function (file) {
      if (file.endsWith("MergeTree.csv")) {
        // Read CSV data
        const mergeTreeData = fs.readFileSync(directoryPath + file, "utf-8");
        console.log(`Processing ${file}`);

        const nodes_data = d3.csvParse(mergeTreeData);
        console.log(nodes_data);

        // process the data

        const nodes_loss = nodes_data.map((d) => {
          return Number(d["Scalar"]);
        });
        console.log(nodes_loss);

        const edgeFile = file.replace("MergeTree", "MergeTree_edge");
        const mergeTreeEdgeData = fs.readFileSync(
          directoryPath + edgeFile,
          "utf-8",
        );
        const edges_data = d3.csvParse(mergeTreeEdgeData);

        const edges = edges_data.map((d) => {
          return [
            Number(d["upNodeId"]),
            Number(d["downNodeId"]),
            Number(d["SegmentationId"]),
          ];
        });

        const segmentationFile = file.replace(
          "MergeTree",
          "MergeTree_segmentation",
        );
        const segmentationData = fs.readFileSync(
          directoryPath + segmentationFile,
          "utf-8",
        );

        const segmentation_data = d3.csvParse(segmentationData);

        const segmentations = segmentation_data.map((d) => {
          return [Number(d["Loss"]), Number(d["SegmentationId"])];
        });

        console.log(edges);

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
        const height = 700;
        const marginTop = 20;
        const marginRight = 20;
        const marginBottom = 30;
        const marginLeft = 40;

        const X_Offset = 50;
        const xScaleMax = 1600;

        class Node {
          constructor(id, loss, totalNumberOfSegmentations, children) {
            this.id = id;
            this.loss = loss;
            this.totalNumberOfSegmentations = totalNumberOfSegmentations;
            this.start = 0;
            this.end = 1600;
            this.children = children;
            this.layoutX = 0;
          }
        }

        const edgeDict = {};
        const loss_extent = d3.extent(nodes_loss);

        // put all segmentations into bins based on segmentationId
        let segmentationCount = {};

        segmentations.forEach((segmentation) => {
          const segmentationId = parseInt(segmentation[1]);
          if (!segmentationCount.hasOwnProperty(segmentationId)) {
            segmentationCount[segmentationId] = 0;
          }
          segmentationCount[segmentationId] += 1;
        });

        // this is used to store number of segmentations based on node id
        // currently, each target node is associated with the segmentation id of an edge
        let total_number_of_segmentations = nodes_loss.map((l) => l);
        edges.forEach((edge) => {
          const segmentationId = parseInt(edge[2]);
          const targetNodeId = edge[1];
          total_number_of_segmentations[targetNodeId] =
            segmentationCount[segmentationId];
        });

        // record the the node id with the largest loss to be the root
        let maxLoss = -1;
        let maxLossNodeId = -1;
        nodes_loss.forEach((item, id) => {
          if (item > maxLoss) {
            maxLoss = item;
            maxLossNodeId = id;
          }
        });

        // assumes maxNode is only one and has one branch?
        // assign the root node with a value, since we missed assigning the root node
        edges.forEach((edge) => {
          if (edge[0] === maxLossNodeId) {
            // total_number_of_segmentations[edge[0]] = total_number_of_segmentations[edge[1]];
            total_number_of_segmentations[edge[0]] = 0;
          }
        });

        // given a nodeId, construct a node
        function getNode(nodeId, nodeDict) {
          let node = null;
          if (nodeDict.hasOwnProperty(nodeId)) {
            node = nodeDict[nodeId];
          } else {
            node = new Node(
              nodeId,
              nodes_loss[nodeId],
              total_number_of_segmentations[nodeId],
              {},
            );
            nodeDict[nodeId] = node;
          }
          return node;
        }

        let nodeDict = {};
        // construct the tree based on edges, node is stored in a dict called nodeDict
        edges.forEach((edge) => {
          const sourceId = edge[0];
          const targetId = edge[1];
          const segmentationId = edge[2];
          let sourceNode = getNode(sourceId, nodeDict);
          let targetNode = getNode(targetId, nodeDict);
          sourceNode.children[targetId] = targetNode;
          edgeDict[segmentationId] = [sourceId, targetId];
        });

        // now, each segmentation also has the loss value. We group them based on segmentation Id
        const groupedSegmentations = {};
        segmentations.forEach((segmentation) => {
          const loss = segmentation[0];
          const segmentationId = segmentation[1];
          if (!groupedSegmentations.hasOwnProperty(segmentationId)) {
            groupedSegmentations[segmentationId] = [];
          }
          groupedSegmentations[segmentationId].push(loss);
        });

        // locate root node
        const root = nodeDict[maxLossNodeId];

        // console.log(total_number_of_segmentations);
        // dfs to just compute the segmentations number
        function dfsComputeTotalSegmentions(root) {
          let thisLevelWidth = 0;
          const children = Object.values(root.children);
          for (const child of children) {
            thisLevelWidth += dfsComputeTotalSegmentions(child);
          }
          const res = thisLevelWidth + total_number_of_segmentations[root.id];
          total_number_of_segmentations[root.id] = thisLevelWidth;
          return res;
        }

        dfsComputeTotalSegmentions(root);

        console.log("acc segmentations");
        console.log(total_number_of_segmentations);

        // traversal of the tree
        // try to assign each node with start value and end value
        function bfs(rootID) {
          // Create a queue to store nodes to be visited
          let queue = [
            {
              root: rootID,
              start: 0,
            },
          ];
          // Create a set to keep track of visited nodes
          const visited = new Set();
          visited.add(rootID);
          // Create an array to store the BFS traversal result
          const result = [];

          // console.log("start");
          // console.log(queue);
          while (queue.length > 0) {
            // Dequeue a node from the queue
            const item = queue.shift();
            const node = nodeDict[item.root];
            const start = item.start;
            // console.log("node is");
            // console.log(node);
            // console.log("start position:");
            // console.log(start);
            // Visit the node
            nodeDict[node.id].start = start;
            nodeDict[node.id].end =
              start + total_number_of_segmentations[node.id];
            result.push(node);

            // Extract child nodes as an array
            let childNodes = Object.values(node.children);
            childNodes = childNodes.sort((a, b) => {
              return (
                total_number_of_segmentations[b.id] -
                total_number_of_segmentations[a.id]
              );
            });

            let left = start;
            console.log(node.id + " child nodes");
            console.log(childNodes.length);
            for (const child of childNodes) {
              if (!visited.has(child.id)) {
                // Enqueue child nodes if not visited
                // console.log("child left");
                // console.log(left);
                const newNode = {
                  root: child.id,
                  start: left,
                };
                queue.push(newNode);
                // console.log("new node");
                // console.log(queue[0]);
                // Mark the child node as visited
                visited.add(child.id);

                left += total_number_of_segmentations[child.id];
                console.log("after left");
                console.log(left);
              }
            }
          }
          return result;
        }

        const bfsTraversal = bfs(root.id);
        // dfs to get the acc segmentations
        // TODO: this is where we need to figure out the acc segmentation and maybe the locations of the node
        // We might not need the bfs
        // function dfs(rootID) {
        //   const root = nodeDict[rootID];
        //   let left = root.start;
        //   let right = root.end;
        //   const children = Object.values(root.children);
        //   for (const child of children) {
        //     const l = total_number_of_segmentations[child.id];
        //     nodeDict[child.id].start = left;
        //     right = left + l;
        //     nodeDict[child.id].end = right;
        //     left = right;
        //     dfs(child.id);
        //   }
        // }

        // dfs(root.id);
        console.log("nodeditc");
        console.log(nodeDict);

        /**
         * DRAWING the plots using D3
         */
        const xScale = d3
          .scaleLinear()
          .domain([0.0001, xScaleMax])
          .range([marginLeft, width - marginRight]);

        const yScale = d3
          .scaleLog()
          .domain(loss_extent)
          .range([height - marginBottom, marginTop]);

        const segmentationScales = {};
        const segmentationPositions = {};

        // Generating the violin shape
        let bins = Object.keys(groupedSegmentations).map((segmentationId) => {
          const segList = groupedSegmentations[segmentationId];

          const bin = d3
            .bin()
            .thresholds(25)
            .value((d) => d)(segList);

          segmentationScales[segmentationId] = d3
            .scaleLinear()
            .domain([bin[0].x0, bin[bin.length - 1].x1])
            .range([yScale(bin[0].x0), yScale(bin[bin.length - 1].x1)]);

          const targetNode = nodeDict[edgeDict[segmentationId][1]];
          segmentationPositions[segmentationId] = targetNode.start;
          return bin;
        });

        // do accumulation of the bins. We might not need this.
        let accBins = [];

        bins.forEach((bin, index) => {
          const accSubBin = [];
          let accLength = 0;
          bin.forEach((b, j) => {
            bins[index][j].segmentationId = index;
            accLength += b.length;
            accSubBin.push(accLength);
          });
          accBins.push(accSubBin);
        });

        accBins = accBins.map((bin, index) => {
          const accSubBin = [];
          bin.forEach((b, j) => {
            accSubBin.push(b + total_number_of_segmentations[index]);
          });
          return accSubBin;
        });

        console.log("acc bins");
        console.log(accBins);
        // Draw
        const svg = body
          .select("#visualization")
          .append("svg")
          .attr("width", width)
          .attr("height", height);

        const areas = svg
          .selectAll(".areas")
          .data(bins)
          .join("g")
          .attr("class", "areas");

        areas
          .selectAll(".area")
          .data((d) => [d])
          .join("path")
          .attr("class", "area")
          .attr("d", (data, binId) => {
            const subsc = segmentationScales[data[0].segmentationId];
            const area = d3
              .area()
              .y(function (_, i) {
                const dd = bins[data[0].segmentationId][i];
                return subsc((dd.x0 + dd.x1) / 2);
              })
              .x0(function (_, i) {
                const currSegmentationId = data[0].segmentationId;
                const dd = accBins[currSegmentationId][i];
                const targetNode = nodeDict[edgeDict[currSegmentationId][1]];

                return xScale(segmentationPositions[currSegmentationId]);
              })
              .x1(function (_, i) {
                const currSegmentationId = data[0].segmentationId;
                const dd = accBins[currSegmentationId][i];
                const targetNode = nodeDict[edgeDict[currSegmentationId][1]];

                return xScale(segmentationPositions[currSegmentationId] + dd);
              });

            return area(data);
          })
          .attr("stroke", (data, i) => {
            const currSegmentationId = data[0].segmentationId;
            const targetNode = nodeDict[edgeDict[currSegmentationId][1]];
            if (targetNode.type === "leaf") {
              return "red";
            } else {
              return "blue";
            }
          })
          .attr("fill", (data, i) => {
            const currSegmentationId = data[0].segmentationId;
            const targetNode = nodeDict[edgeDict[currSegmentationId][1]];
            if (targetNode.type === "leaf") {
              return "red";
            } else {
              return "blue";
            }
          })
          .attr("fill-opacity", 0.5);

        // svg
        //   .selectAll(".line")
        //   .data(edges)
        //   .join("line")
        //   .attr("class", "line")
        //   .attr("x1", (d) => xScale(nodeDict[d[0]].start))
        //   .attr("x2", (d) => xScale(nodeDict[d[1]].start))
        //   .attr("y1", (d) => yScale(nodeDict[d[0]].loss))
        //   .attr("y2", (d) => yScale(nodeDict[d[1]].loss))
        //   .attr("stroke", "rgba(78,103,207,1)")
        //   .attr("stroke-width", "1");

        // svg
        //   .selectAll("circle")
        //   .data(bfsTraversal)
        //   .join("circle")
        //   .attr("fill", (d) => {
        //     if (d.type === "main") {
        //       return "#666";
        //     } else {
        //       return "#666";
        //     }
        //   })
        //   .attr("stroke", "#666")
        //   .attr("cx", (d) => xScale(d.start))
        //   .attr("cy", (d) => yScale(d.loss))
        //   .attr("r", 1);

        // svg
        //   .selectAll("text")
        //   .data(bfsTraversal)
        //   .join("text")
        //   .attr("fill", (d) => {
        //     if (d.type === "main") {
        //       return "black";
        //     } else {
        //       return "black";
        //     }
        //   })
        //   .attr("x", (d) => xScale(d.start) - 5)
        //   .attr("y", (d) => yScale(d.loss) + 15)
        //   .text((d) => d.id)
        //   .attr("font-size", 10);

        // Add the x-axis.
        svg
          .append("g")
          .attr("transform", `translate(0,${height - marginBottom})`)
          .call(d3.axisBottom(xScale));

        // Add the y-axis.
        svg
          .append("g")
          .attr("transform", `translate(${marginLeft},0)`)
          .call(d3.axisLeft(yScale));

        svg
          .append("text")
          .attr("x", 15)
          .attr("y", 15)
          .attr("text-anchor", "start")
          .text("Loss");
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
