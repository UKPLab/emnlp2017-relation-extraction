/*
 *  # coding: utf-8
 *  # Copyright (C) 2017.  UKP lab
 *  #
 *  # Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
 *  # Licensed under Apache License Version 2.0
 */

var TheApp = {};

TheApp.init =  function() {
    $("#getanswerbutton").click(function(){
        $(".resultsrow").find(".panel").hide();
        TheApp.interface.before_query();
        var question_text = $("#questionField").val();
        TheApp.request.current_request.inputtext = question_text;
        /* Clear the last response */
        TheApp.response.last_response = $.extend({}, TheApp.request.current_request);

        TheApp.request.post_input_text(TheApp.request.current_request)
            .then(TheApp.interface.process_results)
            .done(function(){TheApp.interface.after_query();});
    });

    TheApp.draw.vvis = d3.select("#answerField svg");
    TheApp.draw.vvis = TheApp.draw.vvis.append("g").attr("transform", "translate(" + TheApp.draw.vvismargin.left + "," + TheApp.draw.vvismargin.top + ")")
};

/* Requests */

TheApp.request = {};
TheApp.response = {};

TheApp.request.current_request = {};

TheApp.response.t0 = 0;
TheApp.response.t1 = 0;
TheApp.response.last_response = {};

TheApp.request.post = function(url, postdata){
    return $.ajax({
        url: url,
        type: 'POST',
        error: function(){
            $("#answerRow").find(".panel-danger").show();
            TheApp.interface.after_query();
        },
        complete: function(data){
        },
        data: JSON.stringify(postdata),
        dataType: "json",
        processData: false,
        cache: false,
        contentType: "application/json"
    });
};

TheApp.request.post_input_text = function(question_text) {
    return TheApp.request.post("/relation-extraction/parse/", question_text);
};

/* Updating interface */

TheApp.interface = {};

TheApp.interface.before_any_query = function(){
    $(".btn").attr("disabled", "disabled");
};

TheApp.interface.before_query = function(){
    TheApp.interface.before_any_query();
    var answer_row = $("#answerRow");
    answer_row.find(".panel").hide();
    var progress_panel = $("#progressRow");
    progress_panel.find(".progress-bar").width("10%");
    progress_panel.find(".progress").show();
    $("#statisticsRow").find(".panel")
        .show()
        .find("#statsField").empty();
    TheApp.response.t0 = performance.now();
};

TheApp.interface.after_any_query = function(){
    $(".btn").removeAttr("disabled");
};

TheApp.interface.after_query = function(){
    $("#progressRow").find(".progress-bar").width("100%")
        .parent().hide();
    $("#getanswerbutton").removeAttr("disabled");
    TheApp.response.t1 = performance.now();
    $("#statsField").append($("<span />", {'text': "Processing time: " + ((TheApp.response.t1 - TheApp.response.t0) / 1000).toFixed(2) + " seconds"}));
};

TheApp.interface.process_results = function(response) {

};


TheApp.interface.process_results = function(response){
    TheApp.response.last_response.parse = response.relation_graph;
    $("#answerRow").find(".panel-success").show();
    TheApp.draw.draw_graph(response.relation_graph);
};


/* Drawing */
TheApp.draw = {};
TheApp.draw.vvis = {};
TheApp.draw.vvismargin = {top: 20, right: 20, bottom: 30, left: 40};

TheApp.draw.draw_graph = function(s_g){
    var gvis = TheApp.draw.vvis;

    gvis.selectAll("*").remove();
    var nodes2idx = {};
    var nodes = [
    ];
    var links = [];

    for (var edge_i in s_g.edgeSet){
        var edge = s_g.edgeSet[edge_i];
        if (!(edge.left[0] in nodes2idx)){
            nodes2idx[edge.left[0]] = Object.keys(nodes2idx).length;
            nodes.push({ x:   0, y: 0, name: edge.left.map(function (el) {
                return s_g.tokens[el]
            }).join(" ") , class: "node"});
        }
        if (!(edge.right[0] in nodes2idx)){
            nodes2idx[edge.right[0]] = Object.keys(nodes2idx).length;
            nodes.push({ x:   0, y: 0, name: edge.right.map(function (el) {
                return s_g.tokens[el]
            }).join(" "), class: "node"});
        }

        links.push({ source: nodes2idx[edge.left[0]], target: nodes2idx[edge.right[0]],
            name: (edge.kbID !== "P0" ) ? edge.kbID + ":" + edge.lexicalInput : "None", kbID: edge.kbID});

    }

    var link = gvis.selectAll('.link')
        .data(links)
        .enter().append('line')
        .attr('class', function(d) { return (d.kbID !== "P0" ? "link" : "link-dummy"); });

    var width = $("#answerRow").find("svg").innerWidth(), height = $("#answerRow").find("svg").innerHeight();
    var force = d3.forceSimulation()
        .force("charge", d3.forceManyBody().strength(50))
        .force('centerX', d3.forceX(width / 3))
        .force('centerY', d3.forceY(height / 2));

    var node = gvis.selectAll('circle')
        .data(nodes)
        .enter().append('circle')
        .attr('class', function(d) { return d.class; })
        .attr('r', 15)
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    var text = gvis.selectAll(".node-label")
        .data(nodes).enter()
        .append("g")
        .attr('class', "node-label");
    text.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", function(d) { return (d.name.length + 1) + "ex"; })
        .attr("height", 22)
        .attr("rx", 10)
        .attr("rz", 10);
    text.append("text")
        .attr("x", 4)
        .attr("y", 16)
        .text(function(d) { return d.name; });

    var reltext = gvis.selectAll(".rel-label")
        .data(links).enter()
        .append("g")
        .filter(function(d) { return d.kbID !== "P0"})
        .attr('class', "rel-label");
    reltext.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", function(d) { return (d.name.length + 1) + "ex"; })
        .attr("height", 22)
        .attr("rx", 5)
        .attr("rz", 5);
    reltext.append("text")
        .attr("x", 4)
        .attr("y", 16)
        .text(function(d) { return d.name; });

    function tick() {
        link.attr('x1', function(d) { return d.source.x; })
            .attr('y1', function(d) { return d.source.y; })
            .attr('x2', function(d) { return d.target.x; })
            .attr('y2', function(d) { return d.target.y; });
        node.attr("transform", transform);
        text.attr("transform", transform);
        reltext.attr("transform", function(d){ return "translate(" + ((d.source.x + d.target.x)/2 -11) + "," + ((d.source.y + d.target.y)/2 -11) + ")"; } );
    }

    function transform(d) {
        return "translate(" + d.x + "," + d.y + ")";
    }

    force.nodes(nodes).on("tick", tick);
    force.force("link", d3.forceLink(links));
    force.force("link").distance(function(d) { return (d.kbID !== "P0") ? 250 : 250 ; });


    function dragstarted(d) {
        if (!d3.event.active) force.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(d) {
        d.fx = d3.event.x;
        d.fy = d3.event.y;
    }

    function dragended(d) {
        if (!d3.event.active) force.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
};
