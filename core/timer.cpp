#include <iostream>
#include <string>
#include <map>
#include <algorithm>

#include "occa.hpp"
#include "mpi.h"
#include "timer.hpp"

namespace timer {

namespace {

typedef struct tagData_{
  int count;
  double hostElapsed;
  double deviceElapsed;
  double startTime;
  double stopTime;
  occa::streamTag startTag;
  occa::streamTag stopTag;
} tagData;
std::map<std::string,tagData> m_;

const int NEKRS_TIMER_INVALID_KEY    = -1;
const int NEKRS_TIMER_INVALID_METRIC = -2;

int ifSync_;
inline int ifSync(){ return ifSync_; }

occa::device device_;
MPI_Comm comm_;
}

void init(MPI_Comm comm,occa::device device,int ifSync){
  device_=device;
  ifSync_=ifSync;
  comm_ = comm;
}

void reset(){
  m_.clear();
}

void reset(const std::string tag){
  std::map<std::string,tagData>::iterator it=m_.find(tag);
  if(it!=m_.end()) m_.erase(it);
}

void finalize(){
  reset();
}

void deviceTic(const std::string tag,int ifSync){
  if(ifSync) MPI_Barrier(comm_);
  m_[tag].startTag =device_.tagStream();
}

void deviceTic(const std::string tag){
  if(ifSync()) MPI_Barrier(comm_);
  m_[tag].startTag =device_.tagStream();
}

void deviceToc(const std::string tag){
  m_[tag].stopTag =device_.tagStream();

  std::map<std::string,tagData>::iterator it=m_.find(tag);
  if(it==m_.end()){
    printf("Error in deviceToc: Invalid tag name. %s:%u\n",__FILE__,__LINE__);
    MPI_Abort(comm_,1);
  }
}

void hostTic(const std::string tag,int ifSync){
  if(ifSync) MPI_Barrier(comm_);
  m_[tag].startTime=MPI_Wtime();
}

void hostTic(const std::string tag){
  if(ifSync()) MPI_Barrier(comm_);
  m_[tag].startTime=MPI_Wtime();
}

void hostToc(const std::string tag){
  m_[tag].stopTime=MPI_Wtime();

  auto it=m_.find(tag);
  if(it==m_.end()){
    printf("Error in deviceToc: Invalid tag name. %s:%u\n",__FILE__,__LINE__);
    MPI_Abort(comm_,1);
  }
}

void tic(const std::string tag,int ifSync){
  if(ifSync) MPI_Barrier(comm_);
  m_[tag].startTime=MPI_Wtime();
  m_[tag].startTag =device_.tagStream();
}

void tic(const std::string tag){
  if(ifSync()) MPI_Barrier(comm_);
  m_[tag].startTime=MPI_Wtime();
  m_[tag].startTag =device_.tagStream();
}

void toc(const std::string tag){
  m_[tag].stopTime=MPI_Wtime();
  m_[tag].stopTag =device_.tagStream();

  auto it=m_.find(tag);
  if(it==m_.end()){
    printf("Error in deviceToc: Invalid tag name. %s:%u\n",__FILE__,__LINE__);
    MPI_Abort(comm_,1);
  }
}

void update(){
  for (auto it = m_.begin(); it != m_.end(); it++) {
    it->second.hostElapsed += (it->second.stopTime - it->second.startTime);
    it->second.deviceElapsed += device_.timeBetween(it->second.startTag,it->second.stopTag);
    it->second.count++;
  }
}

void update(const std::string tag){
  auto it=m_.find(tag);
  it->second.hostElapsed += (it->second.stopTime - it->second.startTime);
  it->second.deviceElapsed += device_.timeBetween(it->second.startTag,it->second.stopTag);
  it->second.count++;
}

double hostElapsed(const std::string tag){
  auto it=m_.find(tag);
  it->second.hostElapsed = it->second.stopTime - it->second.startTime;
  if(it==m_.end()){ return NEKRS_TIMER_INVALID_KEY; }
  return it->second.hostElapsed;
}

double deviceElapsed(const std::string tag){
  auto it=m_.find(tag);
  if(it==m_.end()){ return NEKRS_TIMER_INVALID_KEY; }
  return it->second.deviceElapsed;
}

int count(const std::string tag){
  auto it=m_.find(tag);
  if(it==m_.end()){ return NEKRS_TIMER_INVALID_KEY; }
  return it->second.count;
}

double query(const std::string tag,const std::string metric){
  int size;
  MPI_Comm_size(comm_,&size);

  auto it=m_.find(tag);
  if(it==m_.end()){ return NEKRS_TIMER_INVALID_KEY; }
  auto hostElapsed=it->second.hostElapsed;
  auto deviceElapsed=it->second.deviceElapsed;
  auto count=it->second.count;

  double retVal;

  std::string upperMetric=metric;
  std::transform(upperMetric.begin(),upperMetric.end(),upperMetric.begin(),::toupper);

  if(upperMetric.compare("HOST:MIN"  )==0) {
    MPI_Allreduce(&hostElapsed,&retVal,1,MPI_DOUBLE,MPI_MIN,comm_);
    return retVal;
  }
  if(upperMetric.compare("HOST:MAX"  )==0) {
    MPI_Allreduce(&hostElapsed,&retVal,1,MPI_DOUBLE,MPI_MAX,comm_);
    return retVal;
  }
  if(upperMetric.compare("HOST:SUM"  )==0) {
    MPI_Allreduce(&hostElapsed,&retVal,1,MPI_DOUBLE,MPI_SUM,comm_);
    return retVal;
  }
  if(upperMetric.compare("HOST:AVG"  )==0) {
    MPI_Allreduce(&hostElapsed,&retVal,1,MPI_DOUBLE,MPI_SUM,comm_);
    return retVal/(size*count);
  }
  if(upperMetric.compare("DEVICE:MIN")==0) {
    MPI_Allreduce(&deviceElapsed,&retVal,1,MPI_DOUBLE,MPI_MIN,comm_);
    return retVal;
  }
  if(upperMetric.compare("DEVICE:MAX")==0) {
    MPI_Allreduce(&deviceElapsed,&retVal,1,MPI_DOUBLE,MPI_MAX,comm_);
    return retVal;
  }
  if(upperMetric.compare("DEVICE:SUM")==0) {
    MPI_Allreduce(&deviceElapsed,&retVal,1,MPI_DOUBLE,MPI_SUM,comm_);
    return retVal;
  }
  if(upperMetric.compare("DEVICE:AVG")==0) {
    MPI_Allreduce(&deviceElapsed,&retVal,1,MPI_DOUBLE,MPI_SUM,comm_);
    return retVal/(size*count);
  }
  return NEKRS_TIMER_INVALID_METRIC;
}


} // namespace
