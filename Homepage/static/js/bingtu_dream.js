
        var dom = document.getElementById('container');
        var myChart = echarts.init(dom, null, {
        renderer: 'canvas',
        useDirtyRect: false
        });
        var app = {};
        
        var option;

        var data = [
    {
        name: 'Video\n26.1M',
        itemStyle: {
        color: '#6495ed'
        },
        children:[
            {
                name: 'Open-sourced \n 7.42M',
                value: 3,
                itemStyle: {
                color: '#bcdff7' 
                },
                children: [
                    {
                        name: 'Academic \n QA \n 503K',
                        value: 1,
                        itemStyle: {
                        color: '#e3f1fa'
                        }
                    },
                    {
                        name: 'Action \n Recognization \n 1.92M', 
                        value: 1,
                        itemStyle: {
                        color: '#cce6f8'
                        }
                    },
                    {
                        name: 'Short \n Caption \n 5.00M',
                        value: 1,
                        itemStyle: {
                        color: '#bcdff7'
                        }
                    }
                ]
            },
            {
                name: 'In-house \n 18.7M', 
                value: 8,
                itemStyle: {
                color: '#87cefa'
                },
                children: [
                    {
                        name: 'GPT4o-labeled \n QA \n 16.2K',
                        value: 1,
                        itemStyle: {
                        color: '#b5ddf7'
                        }
                    },
                    {
                        name: 'Stock-crawled \n Caption \n 2.41M',
                        value: 1,
                        itemStyle: {
                        color: '#afdbf8'
                        }
                    },
                    {
                        name: 'GPT4o-labeled \n Caption \n 2.52M',
                        value: 1,
                        itemStyle: {
                        color: '#a8d9f8'
                        }
                    },
                    {
                        name: 'Human-labeled \n Caption \n 145K',
                        value: 1,
                        itemStyle: {
                        color: '#a2d7f9'
                        }
                    },
                    {
                        name: 'Film & TV \n Commentary \n 11.5M',
                        value: 1,
                        itemStyle: {
                        color: '#9bd4f9'
                        }
                    },
                    {
                        name: 'Object\n Tracking \n 745K',
                        value: 1,
                        itemStyle: {
                        color: '#94d2f9'
                        }
                    },
                    {
                        name: 'Frame \n Permutation \n 825K',
                        value: 1,
                        itemStyle: {
                        color: '#8ed0fa'
                        }
                    },
                    {
                        name: 'Character \n Recognization \n 533K',
                        value: 1,
                        itemStyle: {
                        color: '#87cefa'
                        }
                    },
                    ]
            }
        ]
        
        
    },
    {
        name: 'Image\n11.0M',
        itemStyle: {
        color: '#9899ee'
        },
        children: [
            {
                name: 'Open-sourced \n 9.87M',
                value: 5,
                itemStyle: {
                color: '#cbd4f7'
                },
                children: [
                    {
                        name: 'OCR \n 771K',
                        value: 1,
                        itemStyle: {
                        color: '#b8d2f8'
                        }
                    },
                    {
                        name: 'Academic \n QA \n 799K',
                        value: 1,
                        itemStyle: {
                        color: '#bcd3f8'
                        }
                    },
                    {
                        name: 'Single-image \n Caption \n 5.75M',
                        value: 1,
                        itemStyle: {
                        color: '#c1d3f7'
                        }
                    },
                    {
                        name: 'Multi-image \n QA \n 616K',
                        value: 1,
                        itemStyle: {
                        color: '#c6d4f7'
                        }
                    },
                    {
                        name: 'Multi-image \n Caption \n 1.94M',
                        value: 1,
                        itemStyle: {
                        color: '#cbd4f7'
                        }
                    }
                    ]
            },
            {
                name: 'In-house \n 1.13M',
                value: 1,
                itemStyle: {
                color: '#b2b7f3'
                },
                children: [
                    {
                        name: 'GPT4o-labeled \n Caption \n 1.13M',
                        value: 1,
                        itemStyle: {
                        color: '#b2b7f3'
                        }
                    }
                    ]
            },
        ]



        
    },
    {
        name: 'Text \n 1.11M',
        itemStyle: {
        color: '#84c3b7'
        },
        children: [
        {
            name: 'Open-sourced \n 1.11M',
            value: 1,
            itemStyle: {
            color: '#84c3b7'
            },
            children: [
                {
                    name: 'Instruction \n Tuning \n 1.11M',
                    value: 1,
                    itemStyle: {
                    color: '#84c3b7'
                    }
                }
            ]
        }
        // {
        //     name: 'Basketball',
        //     value: 1,
        //     itemStyle: {
        //     color: '#e7e0eb'
        //     }
        // },
        // {
        //     name: 'Football',
        //     value: 1,
        //     itemStyle: {
        //     color: '#e8dfe6'
        //     }
        // },
        // {
        //     name: 'Athletics',
        //     value: 1,
        //     itemStyle: {
        //     color: '#fce5e9'
        //     }
        // },
        // {
        //     name: 'Other Sports',
        //     value: 1,
        //     itemStyle: {
        //     color: '#efdadf'
        //     }
        // }
        ]
    }
    // {
    //     name: 'Artistic\nPerformance',
    //     itemStyle: {
    //     color: '#f57c6e'
    //     },
    //     children: [
    //     {
    //         name: 'Stage Play',
    //         value: 1,
    //         itemStyle: {
    //         color: '#ffd1d8'
    //         }
    //     },
    //     {
    //         name: 'Magic Show',
    //         value: 1,
    //         itemStyle: {
    //         color: '#f4bec3'
    //         }
    //     },
    //     {
    //         name: 'Variety Show',
    //         value: 1,
    //         itemStyle: {
    //         color: '#ffcbcd'
    //         }
    //     },
    //     {
    //         name: 'Acrobatics',
    //         value: 1,
    //         itemStyle: {
    //         color: '#f9c0b4'
    //         }
    //     }
    //     ]
    // },
    // {
    //     name: 'Life Record',
    //     itemStyle: {
    //     color: '#f2b56f'
    //     },
    //     children: [
    //     {
    //         name: 'Handicraft',
    //         value: 1,
    //         itemStyle: {
    //         color: '#fed5c0'
    //         }
    //     },
    //     {
    //         name: 'Food',
    //         value: 1,
    //         itemStyle: {
    //         color: '#ffcea2'
    //         }
    //     },
    //     {
    //         name: 'Fashion',
    //         value: 1,
    //         itemStyle: {
    //         color: '#faeed3'
    //         }
    //     },
    //     {
    //         name: 'Daily Life',
    //         value: 1,
    //         itemStyle: {
    //         color: '#fedec0'
    //         }
    //     },
    //     {
    //         name: 'Travel',
    //         value: 1,
    //         itemStyle: {
    //         color: '#fbe7d2'
    //         }
    //     },
    //     {
    //         name: 'Pet & Animal',
    //         value: 1,
    //         itemStyle: {
    //         color: '#f8f3da'
    //         }
    //     },
    //     {
    //         name: 'Exercise',
    //         value: 1,
    //         itemStyle: {
    //         color: '#fcecd0'
    //         }
    //     }
    //     ]
    // },
    // {
    //     name: '',
    //     itemStyle: {
    //     color: '#f7e16f'
    //     },
    //     children: [
    //     {
    //         name: 'Multilingual',
    //         value: 1,
    //         itemStyle: {
    //         color: '#fff5c3'
    //         }
    //     }
    //     ]
    // }
    ];
    option = {
    series: {
        type: 'sunburst',
        data: data,
        radius: [0, '95%'],
        sort: undefined,
        emphasis: {
        focus: 'ancestor'
        },
        label: {
        textStyle: {
            // color: "#ffffff",
            fontSize: 12,
            // fontWeight: 'bolder',
            fontFamily: 'serif'
        }
        },
        levels: [
        {},
        {
            r0: '15%',
            r: '40%',
            itemStyle: {
            borderRadius: 6,
            borderWidth: 4
            },
            label: {
            rotate: 'tangential',
            align: 'center',
            fontSize: 16,
            fontWeight: 'bold'
            }
        },
        {
            r0: '40%',
            r: '70%',
            itemStyle: {
            borderRadius: 6,
            borderWidth: 4
            },
            label: {
                // rotate: 'tangential',
                align: 'center',
                fontSize: 14,
                // fontWeight: 'bold'
            }
        },
        {
            r0: '70%',
            r: '72%',
            label: {
            position: 'outside',
            padding: 3,
            silent: false
            },
            itemStyle: {
            // borderWidth: 3
            }
        }
        ]
    }
    };
        if (option && typeof option === 'object') {
        myChart.setOption(option);
        }

window.addEventListener('resize', myChart.resize);