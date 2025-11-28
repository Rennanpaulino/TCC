package com.seutcc.app.adapters;

import android.graphics.Color;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.seutcc.app.R;
import com.seutcc.app.models.Trem;
import java.util.ArrayList;
import java.util.List;

public class TremAdapter extends RecyclerView.Adapter<TremAdapter.TremViewHolder> {

    private List<Trem> listaTrens = new ArrayList<>();
    private OnItemClickListener listener;

    // Interface para lidar com cliques (callback para a MainActivity)
    public interface OnItemClickListener {
        void onItemClick(Trem trem);
    }

    public TremAdapter(OnItemClickListener listener) {
        this.listener = listener;
    }

    public void atualizarDados(List<Trem> novosTrens) {
        this.listaTrens = novosTrens;
        notifyDataSetChanged(); // Avisa a tela que mudou tudo
    }

    @NonNull
    @Override
    public TremViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_trem, parent, false);
        return new TremViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull TremViewHolder holder, int position) {
        Trem trem = listaTrens.get(position);

        holder.txtId.setText(trem.getId());
        holder.txtLotacao.setText("Lotação Estimada: " + trem.getLotacao());

        // LÓGICA DAS CORES (Igual faríamos no mapa)
        if (trem.getLotacao() < 10) {
            holder.viewStatus.setBackgroundColor(Color.parseColor("#4CAF50")); // Verde
        } else if (trem.getLotacao() < 30) {
            holder.viewStatus.setBackgroundColor(Color.parseColor("#FFC107")); // Amarelo
        } else {
            holder.viewStatus.setBackgroundColor(Color.parseColor("#F44336")); // Vermelho
        }

        // Configura o clique
        holder.itemView.setOnClickListener(v -> listener.onItemClick(trem));
    }

    @Override
    public int getItemCount() {
        return listaTrens.size();
    }

    // Classe interna que segura os elementos visuais
    static class TremViewHolder extends RecyclerView.ViewHolder {
        TextView txtId, txtLotacao;
        View viewStatus;

        public TremViewHolder(@NonNull View itemView) {
            super(itemView);
            txtId = itemView.findViewById(R.id.txtIdTrem);
            txtLotacao = itemView.findViewById(R.id.txtLotacao);
            viewStatus = itemView.findViewById(R.id.viewStatus);
        }
    }
}