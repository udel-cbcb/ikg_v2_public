#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <map>
#include <pybind11/stl.h>
#include <iostream>
#include <Eigen/LU>

namespace py = pybind11;
typedef Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatI;

void to_indexed_triples(std::vector<std::tuple<std::string,std::string,std::string>> triples_named,
                        Eigen::Ref<MatI> triples_indexed,
                        std::map<std::string,int64_t> entity_map,
                        std::map<std::string,int64_t> relation_map){
    

    auto num_records = triples_named.size();

    for(auto record_index = 0;record_index < num_records;++record_index){

        auto record = triples_named.at(record_index);

        auto head = std::get<0>(record);
        auto relation = std::get<1>(record);
        auto tail = std::get<2>(record);

        auto head_index = entity_map.at(head);
        auto rel_index = relation_map.at(relation);
        auto tail_index = entity_map.at(tail);

        triples_indexed(record_index,0) = head_index;
        triples_indexed(record_index,1) = rel_index;
        triples_indexed(record_index,2) = tail_index;
    }

}