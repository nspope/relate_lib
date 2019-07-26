#ifndef MUTATIONS_HPP
#define MUTATIONS_HPP

#include "gzstream.hpp"
#include "data.hpp"
#include "anc.hpp"

#include <iostream>
#include <deque>

struct SNPInfo{

  std::string rs_id;
  int snp_id;
  int pos, dist;

  int tree;
  std::deque<int> branch;
  std::vector<int> freq;
  bool flipped = false;
  float age_begin = 0.0, age_end = 0.0;

  std::string upstream_base = "NA", downstream_base = "NA";
  std::string mutation_type = "NA";

};


class Mutations{

  private:

    int N;
    int L;

    int num_flips, num_notmappingmutations;

  public:

    std::string header;
    std::vector<SNPInfo> info;

    Mutations(){};
    Mutations(Data& data);
    void Init(Data& data);

    //////////////////////////////////////////////////////

    void GetAge(AncesTree& anc);
   
    void Read(igzstream& is);
    void Read(const std::string& filename);
    void Dump(const std::string& filename);

    int GetNumFlippedMutations(){return num_flips;}
    int GetNumNotMappingMutations(){return num_notmappingmutations;}

};

class AncMutIterators{

  private:

    igzstream is;
    Muts::iterator pit_mut;
    Mutations mut;

    int N, num_trees;
    int tree_index_in_anc, tree_index_in_mut;
    double num_bases_tree_persists;
    std::string line;

  public:

    AncMutIterators(std::string filename_anc, std::string filename_mut);

    double NextTree(MarginalTree& mtr, Muts::iterator& it_mut);
    double FirstSNP(MarginalTree& mtr, Muts::iterator& it_mut);
    double NextSNP(MarginalTree& mtr, Muts::iterator& it_mut);

};


#endif //MUTATIONS_HPP 
